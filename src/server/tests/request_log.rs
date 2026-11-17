//! The request log through the router (issue #642): which
//! lines the handlers log, in which order, and which of
//! them wait for what.

use super::*;
use crate::observe::{FakeStdout, QUEUE_BYTES};
use std::time::Duration;

/// Long enough for anything that must happen to happen on
/// a loaded machine; a test that passes never waits it out.
const LONG: Duration = Duration::from_secs(60);

/// Wait until `done`, checking every millisecond.
async fn until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + LONG;
    while !done() {
        assert!(
            std::time::Instant::now() < deadline,
            "never happened: {what}"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// A server that logs to `log`, with uploads on in `dir`.
async fn spawn_logging(log: RequestLog, dir: &std::path::Path) -> String {
    let dir = std::fs::canonicalize(dir).unwrap();
    spawn_state(AppState {
        request_log: log,
        loader: Loader::with_defaults()
            .unwrap()
            .with_sources(crate::loader::SourceConfig {
                file_root: Some(dir.clone()),
                ..Default::default()
            }),
        metrics_token: Some(TEST_METRICS_TOKEN.to_string()),
        upload: Some(Arc::new(crate::upload::UploadStore::local(
            dir,
            1024 * 1024,
            false,
        ))),
        ..test_state()
    })
    .await
}

/// One fixed run of requests against `base`, one after the
/// other: the answers' statuses, in order. Eleven of them
/// reach the image route and log a request line, two are
/// purges, one is an upload, and two (the health check and
/// the scrape) log nothing.
async fn replay(base: &str, src: &str, missing: &str) -> Vec<u16> {
    let mut statuses = Vec::new();
    let image = format!("50x30/filters:format(jpeg)/{src}");
    let signed = format!("{base}/{}/{image}", sign(&image));
    // A miss, a hit, a revalidation, a HEAD.
    let resp = client().get(&signed).send().await.unwrap();
    let etag = header(&resp, "etag");
    statuses.push(resp.status().as_u16());
    let resp = client().get(&signed).send().await.unwrap();
    statuses.push(resp.status().as_u16());
    let resp = client()
        .get(&signed)
        .header("If-None-Match", &etag)
        .send()
        .await
        .unwrap();
    statuses.push(resp.status().as_u16());
    let resp = client().head(&signed).send().await.unwrap();
    statuses.push(resp.status().as_u16());
    // The refusals: a bad signature, a filter the grammar
    // does not know, an expired URL, a source that is gone.
    let resp = client()
        .get(format!("{base}/{}/{image}", "0".repeat(64)))
        .send()
        .await
        .unwrap();
    statuses.push(resp.status().as_u16());
    for remainder in [
        format!("filters:nope()/{src}"),
        format!("exp:1/{image}"),
        format!("20x15/{missing}"),
    ] {
        statuses.push(get_signed(base, &remainder).await.status().as_u16());
    }
    // A request ID the client chose, on a hit.
    let resp = get_with_headers(base, &image, &[("X-Request-Id", "chosen-by-client")]).await;
    statuses.push(resp.status().as_u16());
    // Neither of these logs a line.
    let resp = client().get(format!("{base}/health")).send().await.unwrap();
    statuses.push(resp.status().as_u16());
    statuses.push(metrics_response(base).await.status().as_u16());
    // A purge that is refused, one that removes the entry,
    // and the miss that follows it.
    for signature in [
        "deadbeef".to_string(),
        crate::url::purge_signature(SECRET, &image),
    ] {
        let resp = client()
            .post(format!("{base}/purge/{signature}/{image}"))
            .send()
            .await
            .unwrap();
        statuses.push(resp.status().as_u16());
    }
    let resp = client().get(&signed).send().await.unwrap();
    statuses.push(resp.status().as_u16());
    // An upload, and the image served from it.
    let name = "logged/cat.png";
    let resp = client()
        .put(format!(
            "{base}/upload/{}/{name}",
            crate::url::upload_signature(SECRET, name)
        ))
        .body(test_png())
        .send()
        .await
        .unwrap();
    statuses.push(resp.status().as_u16());
    let uploaded = format!("40x24/file:///{name}");
    statuses.push(get_signed(base, &uploaded).await.status().as_u16());
    statuses
}

/// A log's lines with what differs from run to run taken
/// out: the clock fields of a request line, and a request
/// ID the server generated.
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

#[tokio::test]
async fn a_replay_logs_the_same_lines_in_the_same_order_queued_or_inline() {
    let (src, _) = spawn_counting_server(200, "image/png", test_png()).await;
    let (missing, _) = spawn_counting_server(404, "text/plain", b"gone".to_vec()).await;
    let mut logged = Vec::new();
    for (form, out) in [
        ("inline", FakeStdout::default()),
        ("queued", FakeStdout::default()),
    ] {
        let log = match form {
            "inline" => RequestLog::inline_to(out.clone()),
            _ => RequestLog::queued_to(out.clone()),
        };
        let dir = upload_dir(&format!("log-replay-{form}"));
        let base = spawn_logging(log.clone(), &dir).await;
        let statuses = replay(&base, &src, &missing).await;
        assert_eq!(
            statuses,
            [
                200, 200, 304, 200, 403, 400, 403, 404, 200, 200, 200, 403, 200, 200, 201, 200
            ],
            "{form}"
        );
        // A shutdown's last line ends the log, after every
        // line the requests left.
        assert_eq!(log.finish(Some("shutdown: done"), LONG).await, 0, "{form}");
        logged.push(out.lines());
    }
    let (inline, queued) = (comparable(&logged[0]), comparable(&logged[1]));
    assert_eq!(inline, queued);
    // What the lines are, so the two cannot agree on a log
    // that says nothing: one per image request, with its
    // status, in the order of the requests, and the purge
    // and upload lines in their places.
    let records: Vec<serde_json::Value> = queued
        .iter()
        .take(queued.len() - 1)
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let told: Vec<String> = records
        .iter()
        .map(|record| match record.get("event") {
            Some(event) => event.as_str().unwrap().to_string(),
            None => record["status"].to_string(),
        })
        .collect();
    assert_eq!(
        told,
        [
            "200", "200", "304", "200", "403", "400", "403", "404", "200", "purge", "purge", "200",
            "upload", "200"
        ]
    );
    assert_eq!(records[0]["cache"], "miss");
    assert_eq!(records[1]["cache"], "hit");
    // The fourth is the HEAD request, and says so (issue
    // #707).
    assert_eq!(records[2]["method"], "GET");
    assert_eq!(records[3]["method"], "HEAD");
    assert_eq!(records[0]["ops"], "50x30/filters:format(jpeg)");
    assert_eq!(records[8]["request_id"], "chosen-by-client");
    assert_eq!(records[9]["rejected"], true);
    assert_eq!(records[10]["removed"], 1);
    assert_eq!(records[11]["cache"], "miss");
    assert_eq!(records[12]["outcome"], "stored");
    assert_eq!(queued.last().unwrap(), "shutdown: done");
}

#[tokio::test]
async fn a_purge_and_an_upload_are_on_standard_output_before_they_are_answered() {
    let out = FakeStdout::default();
    let (log, _) = RequestLog::over(&out, QUEUE_BYTES);
    let dir = upload_dir("log-audit-first");
    let base = spawn_logging(log, &dir).await;
    // Nobody reads standard output. An image request is
    // answered all the same: its line is in the queue,
    // which has room.
    out.stall();
    let image = format!("{base}/{}/20x15/http://127.0.0.1:9/a.png", "0".repeat(64));
    let resp = tokio::time::timeout(LONG, client().get(&image).send())
        .await
        .expect("an image request waits for the write of its line")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    out.blocked().await;
    // A purge and an upload are not: each waits for its
    // line to be written.
    let purge = tokio::spawn(
        client()
            .post(format!(
                "{base}/purge/deadbeef/20x15/http://127.0.0.1:9/a.png"
            ))
            .send(),
    );
    let name = "audited.png";
    let upload = tokio::spawn(
        client()
            .put(format!(
                "{base}/upload/{}/{name}",
                crate::url::upload_signature(SECRET, name)
            ))
            .body(test_png())
            .send(),
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !purge.is_finished(),
        "the purge was answered before its line"
    );
    assert!(
        !upload.is_finished(),
        "the upload was answered before its line"
    );
    assert_eq!(out.lines().len(), 0);
    // Standard output is read again: both are answered,
    // and their lines are there by then.
    out.resume();
    let purge = tokio::time::timeout(LONG, purge)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(purge.status(), StatusCode::FORBIDDEN);
    assert!(out.text().contains("\"event\":\"purge\""), "{}", out.text());
    let upload = tokio::time::timeout(LONG, upload)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(upload.status(), StatusCode::CREATED);
    assert!(
        out.text().contains("\"event\":\"upload\""),
        "{}",
        out.text()
    );
    let lines = out.lines();
    assert_eq!(lines.len(), 3, "{lines:?}");
    assert!(lines[0].contains("\"status\":403"), "{lines:?}");
}

#[tokio::test]
async fn a_request_waits_while_the_queue_is_full_and_the_server_keeps_running() {
    // A queue that is full at 600 bytes: two request lines
    // and part of a third.
    let out = FakeStdout::default();
    let (log, notes) = RequestLog::over(&out, 600);
    let dir = upload_dir("log-full-queue");
    let base = spawn_logging(log.clone(), &dir).await;
    out.stall();
    let refused = |n: usize| {
        let url = format!("{base}/{}/20x15/http://127.0.0.1:9/a.png", "0".repeat(64));
        client()
            .get(url)
            .header("X-Request-Id", format!("request-{n}"))
            .send()
    };
    // Requests are answered while the queue has room: their
    // lines are queued, with nobody reading.
    let mut answered = 0;
    while !log.stats().full {
        let resp = tokio::time::timeout(LONG, refused(answered))
            .await
            .expect("a request waits though the queue has room")
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        answered += 1;
        assert!(answered < 10, "the queue never filled");
    }
    assert!(answered >= 2, "{answered} lines filled 600 bytes");
    let held = log.stats().held_bytes;
    // The queue is full. The next request is not handled: it
    // waits for room, and so do a purge, a group purge and
    // an upload, which leave lines too. This runtime has one
    // thread: a wait that parked it would stop the test
    // here, or at the health check below.
    let waiting = tokio::spawn(refused(answered));
    until("the request waits for room", || log.stats().waiting == 1).await;
    let purge = tokio::spawn(
        client()
            .post(format!(
                "{base}/purge/deadbeef/20x15/http://127.0.0.1:9/a.png"
            ))
            .send(),
    );
    until("the purge waits for room", || log.stats().waiting == 2).await;
    let group_purge = tokio::spawn(
        client()
            .post(format!("{base}/purge-group/deadbeef/heroes"))
            .send(),
    );
    until("the group purge waits for room", || {
        log.stats().waiting == 3
    })
    .await;
    let name = "waited.png";
    let upload = tokio::spawn(
        client()
            .put(format!(
                "{base}/upload/{}/{name}",
                crate::url::upload_signature(SECRET, name)
            ))
            .body(test_png())
            .send(),
    );
    until("the upload waits for room", || log.stats().waiting == 4).await;
    // Clients that give up while their requests wait take
    // the requests with them: they were never handled, so
    // they leave no line, and the queue holds nothing for
    // them, however many come and go.
    for n in 0..5 {
        let gone = tokio::spawn(refused(900 + n));
        until("one more request waits", || log.stats().waiting == 5).await;
        gone.abort();
        until("it left with its client", || log.stats().waiting == 4).await;
    }
    assert_eq!(log.stats().held_bytes, held);
    // The rest of the server does not wait: a request that
    // logs nothing is answered.
    let health = tokio::time::timeout(LONG, client().get(format!("{base}/health")).send())
        .await
        .expect("the health check waits behind a full log queue")
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert!(!waiting.is_finished());
    // Standard output is read again: the four are handled
    // and answered. The log holds the line of every request
    // that was answered, once, the earlier ones in their
    // order, and none for the clients that left.
    out.resume();
    for (request, status) in [
        (waiting, StatusCode::FORBIDDEN),
        (purge, StatusCode::FORBIDDEN),
        (group_purge, StatusCode::FORBIDDEN),
        (upload, StatusCode::CREATED),
    ] {
        let resp = tokio::time::timeout(LONG, request)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(resp.status(), status);
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    let records: Vec<serde_json::Value> = out
        .lines()
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let ids: Vec<&str> = records
        .iter()
        .filter_map(|record| record["request_id"].as_str())
        .collect();
    let expected: Vec<String> = (0..=answered).map(|n| format!("request-{n}")).collect();
    assert_eq!(ids, expected);
    let mut events: Vec<&str> = records
        .iter()
        .filter_map(|record| record["event"].as_str())
        .collect();
    events.sort();
    assert_eq!(events, ["purge", "purge-group", "upload"]);
    assert_eq!(records.len(), answered + 4);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_served_at_once_each_leave_their_line_once() {
    // A queue that fills over and over, and a standard
    // output that takes a few bytes per write: handlers on
    // four threads append while the writer is busy, and
    // requests wait for room all the time.
    let out = FakeStdout::default();
    let (log, notes) = RequestLog::over(&out, 4096);
    out.take_at_most(61);
    let dir = upload_dir("log-at-once");
    let base = spawn_logging(log.clone(), &dir).await;
    const CLIENTS: usize = 16;
    const EACH: usize = 25;
    let mut clients = Vec::new();
    for c in 0..CLIENTS {
        let base = base.clone();
        clients.push(tokio::spawn(async move {
            // One connection for each client: it sends a
            // request once the one before is answered.
            let client = client();
            for n in 0..EACH {
                let resp = client
                    .get(format!(
                        "{base}/{}/20x15/http://127.0.0.1:9/a.png",
                        "0".repeat(64)
                    ))
                    .header("X-Request-Id", format!("client-{c}-request-{n}"))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::FORBIDDEN);
            }
        }));
    }
    for client in clients {
        tokio::time::timeout(LONG, client).await.unwrap().unwrap();
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    // Every request left its line, whole and once, and each
    // client's lines are in the order of its requests.
    let mut next = [0usize; CLIENTS];
    for line in out.lines() {
        let record: serde_json::Value =
            serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
        let id = record["request_id"].as_str().unwrap();
        let numbers: Vec<usize> = id.split('-').filter_map(|part| part.parse().ok()).collect();
        let [c, n] = numbers[..] else {
            panic!("not a request of this test: {line}")
        };
        assert_eq!(n, next[c], "{line}");
        next[c] += 1;
    }
    assert_eq!(next, [EACH; CLIENTS]);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

/// An image request to `base` that is refused at once, with
/// request ID `id`: its line is a few hundred bytes.
async fn refused_request(base: String, id: String) -> reqwest::Response {
    let url = format!("{base}/{}/20x15/http://127.0.0.1:9/a.png", "0".repeat(64));
    tokio::time::timeout(LONG, client().get(url).header("X-Request-Id", id).send())
        .await
        .expect("an answer")
        .unwrap()
}

/// Send refused requests to `base` while nobody reads `log`,
/// one after the other, until the queue is full: how many
/// were answered, and when the one that filled it was sent,
/// which is no later than the queue filled.
async fn fill_queue(base: &str, log: &RequestLog) -> (usize, std::time::Instant) {
    let mut answered = 0;
    let mut filled_by = std::time::Instant::now();
    while !log.stats().full {
        filled_by = std::time::Instant::now();
        let resp = refused_request(base.to_string(), format!("request-{answered}")).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        answered += 1;
        assert!(answered < 10, "the queue never filled");
    }
    (answered, filled_by)
}

/// The value of the metric `name` in `exposition`.
fn metric_value(exposition: &str, name: &str) -> f64 {
    exposition
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
        .unwrap_or_else(|| panic!("no {name} in:\n{exposition}"))
}

#[tokio::test]
async fn the_health_check_fails_while_the_queue_stays_full_and_passes_again_after() {
    // Issue #707: with the queue full every request that
    // logs waits, so the server answers none of them, and
    // the health check still answered 200: a liveness
    // probe never restarted a replica in that state. Now it
    // answers 503 once the queue has stayed full for the
    // log's bound (5 s in the server; 1 s here), without
    // ever waiting on the queue, and 200 again once
    // standard output is read. `/metrics` shows the queue.
    let bound = Duration::from_secs(1);
    let out = FakeStdout::default();
    let (log, notes) = RequestLog::over_stalling(&out, 600, bound);
    let dir = upload_dir("log-stalled-health");
    let base = spawn_logging(log.clone(), &dir).await;
    let health = |accept: &'static str| {
        let url = format!("{base}/health");
        async move {
            tokio::time::timeout(LONG, client().get(url).header("Accept", accept).send())
                .await
                .expect("the health check waits on the queue")
                .unwrap()
        }
    };
    let resp = health("*/*").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.text().await.unwrap(), "ok");
    out.stall();
    let (answered, filled_by) = fill_queue(&base, &log).await;
    // Full, but not for long yet: the check passes, and
    // the queue holds its cap. (On a machine slow enough
    // to spend the bound on the request, it may already
    // fail, which the loop below expects anyway.)
    let early = health("*/*").await.status();
    if filled_by.elapsed() < bound {
        assert_eq!(early, StatusCode::OK);
    }
    let metrics = metrics_text(&base).await;
    assert!(metric_value(&metrics, "crabcrop_request_log_queue_bytes") >= 600.0);
    assert_eq!(
        metric_value(&metrics, "crabcrop_request_log_queue_capacity_bytes"),
        600.0
    );
    assert_eq!(
        metric_value(&metrics, "crabcrop_request_log_requests_waiting"),
        0.0
    );
    // The next request waits for room.
    let waiting = tokio::spawn(refused_request(base.clone(), "waited".to_string()));
    until("the request waits for room", || log.stats().waiting == 1).await;
    // The check answers at once all along, and fails once
    // the queue has been full for the bound.
    let stalled = loop {
        let resp = health("application/json").await;
        if resp.status() == StatusCode::SERVICE_UNAVAILABLE {
            break resp;
        }
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(filled_by.elapsed() < LONG, "the health check never failed");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert!(
        filled_by.elapsed() >= bound,
        "it failed {:?} after the queue filled",
        filled_by.elapsed()
    );
    assert_eq!(header(&stalled, "cache-control"), "no-store");
    let body: serde_json::Value = serde_json::from_str(&stalled.text().await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "request_log_stalled");
    let plain = health("*/*").await;
    assert_eq!(plain.status(), StatusCode::SERVICE_UNAVAILABLE);
    let text = plain.text().await.unwrap();
    assert!(
        text.starts_with("request log stalled: its queue has been full for "),
        "{text}"
    );
    let head = client()
        .head(format!("{base}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(head.status(), StatusCode::SERVICE_UNAVAILABLE);
    let metrics = metrics_text(&base).await;
    assert_eq!(
        metric_value(&metrics, "crabcrop_request_log_requests_waiting"),
        1.0
    );
    assert!(
        metric_value(&metrics, "crabcrop_request_log_queue_full_seconds_total")
            >= bound.as_secs_f64(),
        "{metrics}"
    );
    // The request still waits: nothing was dropped.
    assert!(!waiting.is_finished());
    // Standard output is read again: the request is
    // answered, the check passes, and the time spent full
    // stops growing.
    out.resume();
    let resp = tokio::time::timeout(LONG, waiting).await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    until("the queue has room again", || !log.stats().full).await;
    assert_eq!(health("*/*").await.status(), StatusCode::OK);
    let metrics = metrics_text(&base).await;
    assert_eq!(
        metric_value(&metrics, "crabcrop_request_log_requests_waiting"),
        0.0
    );
    let spent = metric_value(&metrics, "crabcrop_request_log_queue_full_seconds_total");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let metrics = metrics_text(&base).await;
    assert_eq!(
        metric_value(&metrics, "crabcrop_request_log_queue_full_seconds_total"),
        spent
    );
    // Every line is there, the waiting request's included.
    assert_eq!(log.finish(None, LONG).await, 0);
    let ids: Vec<String> = out
        .lines()
        .iter()
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            record["request_id"].as_str().unwrap().to_string()
        })
        .collect();
    let mut expected: Vec<String> = (0..answered).map(|n| format!("request-{n}")).collect();
    expected.push("waited".to_string());
    assert_eq!(ids, expected);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test]
async fn a_request_that_waited_for_room_counts_the_wait_in_its_time() {
    // Issue #707: the request's timer started after its
    // wait for room, so neither its line nor the latency
    // metrics showed the wait.
    let out = FakeStdout::default();
    let (log, _) = RequestLog::over(&out, 600);
    let dir = upload_dir("log-wait-timed");
    let base = spawn_logging(log.clone(), &dir).await;
    out.stall();
    fill_queue(&base, &log).await;
    let waiting = tokio::spawn(refused_request(base.clone(), "waited".to_string()));
    until("the request waits for room", || log.stats().waiting == 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    out.resume();
    let resp = tokio::time::timeout(LONG, waiting).await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(log.finish(None, LONG).await, 0);
    let waited: serde_json::Value = out
        .lines()
        .iter()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|record| record["request_id"] == "waited")
        .expect("the waiting request's line");
    let total = waited["total_ms"].as_f64().unwrap();
    assert!(total >= 300.0, "{waited}");
    let metrics = metrics_text(&base).await;
    assert!(
        metric_value(&metrics, "crabcrop_request_duration_ms_sum") >= 300.0,
        "{metrics}"
    );
    assert!(
        metric_value(&metrics, "crabcrop_request_duration_ms_bucket{le=\"250\"}")
            < metric_value(&metrics, "crabcrop_request_duration_ms_bucket{le=\"+Inf\"}"),
        "{metrics}"
    );
}

/// A record of an image request that got as far as a render,
/// owed to a fresh state whose request log writes inline to a
/// stand-in for standard output.
fn owed() -> (OwedRecord, Arc<AppState>, FakeStdout) {
    let out = FakeStdout::default();
    let state = Arc::new(AppState {
        request_log: RequestLog::inline_to(out.clone()),
        ..test_state()
    });
    let record = RequestRecord {
        ops: "300x50/filters:format(jxl)".to_string(),
        load: Duration::from_millis(3),
        process: Duration::from_millis(1),
        load_ran: true,
        process_ran: true,
        cache_looked_up: true,
        bytes_in: 1200,
        bytes_out: 300,
        request_id: "owed".to_string(),
        ..RequestRecord::default()
    };
    let owed = OwedRecord::new(record, std::time::Instant::now(), Arc::clone(&state));
    (owed, state, out)
}

/// The lines `out` holds, parsed.
fn lines_of(out: &FakeStdout) -> Vec<serde_json::Value> {
    out.lines()
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn a_record_filed_with_its_answer_is_recorded_once() {
    let (owed, state, out) = owed();
    owed.file(304);
    let lines = lines_of(&out);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["status"], 304);
    assert_eq!(lines[0]["bytes_out"], 300);
    assert_eq!(lines[0]["request_id"], "owed");
    let exposition = state.metrics.render();
    assert!(exposition.contains("crabcrop_requests_total{status_class=\"3xx\"} 1\n"));
    assert!(exposition.contains("crabcrop_request_duration_ms_count 1\n"));
}

#[test]
fn a_record_dropped_unfiled_is_recorded_as_client_closed() {
    // Issue #707: a request whose client left was dropped
    // with its record, and left neither a line nor a metrics
    // update. Its record now files itself.
    let (owed, state, out) = owed();
    std::thread::sleep(Duration::from_millis(20));
    drop(owed);
    let lines = lines_of(&out);
    assert_eq!(lines.len(), 1);
    let line = &lines[0];
    assert_eq!(line["status"], crate::observe::CLIENT_CLOSED);
    // What it got through is kept, and it served nothing.
    assert_eq!(line["ops"], "300x50/filters:format(jxl)");
    assert_eq!(line["load_ms"], 3.0);
    assert_eq!(line["bytes_in"], 1200);
    assert_eq!(line["bytes_out"], 0);
    assert_eq!(line["request_id"], "owed");
    // Its time runs to the drop.
    assert!(line["total_ms"].as_f64().unwrap() >= 20.0, "{line}");
    let exposition = state.metrics.render();
    assert!(exposition.contains("crabcrop_requests_total{status_class=\"4xx\"} 1\n"));
    assert!(exposition.contains("crabcrop_bytes_served_total 0\n"));
    assert!(exposition.contains("crabcrop_cache_misses_total 1\n"));
}

#[test]
fn a_record_dropped_in_a_panic_or_owed_nowhere_records_nothing() {
    let (owed, state, out) = owed();
    // Unwinds like a panic, without the message a panic
    // prints.
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _owed = owed;
        std::panic::resume_unwind(Box::new("a request that panics"));
    }));
    assert!(unwound.is_err());
    assert!(out.lines().is_empty());
    assert!(
        state
            .metrics
            .render()
            .contains("crabcrop_request_duration_ms_count 0\n")
    );
    // A settled record files nothing either way.
    OwedRecord::settled().file(200);
    drop(OwedRecord::settled());
}
