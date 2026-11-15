//! The request log through the router (issue #642): which
//! lines the handlers log, in which order, and which of
//! them wait for what.

use super::*;
use crate::observe::{FakeStdout, QUEUE_BYTES};
use std::time::Duration;

/// Long enough for anything that must happen; a test that
/// passes never waits it out.
const LONG: Duration = Duration::from_secs(30);

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
    for path in ["/health", "/metrics"] {
        let resp = client().get(format!("{base}{path}")).send().await.unwrap();
        statuses.push(resp.status().as_u16());
    }
    // A purge that is refused, one that removes the entry,
    // and the miss that follows it.
    for signature in ["deadbeef".to_string(), crate::url::purge_signature(SECRET, &image)] {
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
                200, 200, 304, 200, 403, 400, 410, 404, 200, 200, 200, 403, 200, 200, 201, 200
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
            "200", "200", "304", "200", "403", "400", "410", "404", "200", "purge", "purge", "200",
            "upload", "200"
        ]
    );
    assert_eq!(records[0]["cache"], "miss");
    assert_eq!(records[1]["cache"], "hit");
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
    // answered all the same: its line is in the queue.
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
            .post(format!("{base}/purge/deadbeef/20x15/http://127.0.0.1:9/a.png"))
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
    assert!(!purge.is_finished(), "the purge was answered before its line");
    assert!(!upload.is_finished(), "the upload was answered before its line");
    assert_eq!(out.lines().len(), 0);
    // Standard output is read again: both are answered,
    // and their lines are there by then.
    out.resume();
    let purge = tokio::time::timeout(LONG, purge).await.unwrap().unwrap().unwrap();
    assert_eq!(purge.status(), StatusCode::FORBIDDEN);
    assert!(out.text().contains("\"event\":\"purge\""), "{}", out.text());
    let upload = tokio::time::timeout(LONG, upload).await.unwrap().unwrap().unwrap();
    assert_eq!(upload.status(), StatusCode::CREATED);
    assert!(out.text().contains("\"event\":\"upload\""), "{}", out.text());
    let lines = out.lines();
    assert_eq!(lines.len(), 3, "{lines:?}");
    assert!(lines[0].contains("\"status\":403"), "{lines:?}");
}

#[tokio::test]
async fn a_request_waits_for_room_in_a_full_queue_and_the_server_keeps_running() {
    // A queue of 600 bytes: two request lines and part of a
    // third.
    let out = FakeStdout::default();
    let (log, notes) = RequestLog::over(&out, 600);
    let dir = upload_dir("log-full-queue");
    let base = spawn_logging(log.clone(), &dir).await;
    out.stall();
    let refused = |n: usize| {
        let url = format!(
            "{base}/{}/20x15/http://127.0.0.1:9/a.png",
            "0".repeat(64)
        );
        client()
            .get(url)
            .header("X-Request-Id", format!("request-{n}"))
            .send()
    };
    // Requests are answered while their lines fit. This
    // runtime has one thread: a wait that parked it would
    // stop the test here, or at the health check below.
    let mut answered = 0;
    let waiting = loop {
        let mut request = tokio::spawn(refused(answered));
        match tokio::time::timeout(Duration::from_millis(500), &mut request).await {
            Ok(resp) => {
                assert_eq!(resp.unwrap().unwrap().status(), StatusCode::FORBIDDEN);
                answered += 1;
                assert!(answered < 10, "the queue never filled");
            }
            Err(_) => break request,
        }
    };
    assert!(answered >= 2, "{answered} lines fit in 600 bytes");
    // That one's line found the queue full: its response
    // waits. The rest of the server does not: a request
    // that logs nothing is answered.
    let health = tokio::time::timeout(LONG, client().get(format!("{base}/health")).send())
        .await
        .expect("the health check waits behind a full log queue")
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert!(!waiting.is_finished());
    // Standard output is read again: the request is
    // answered, and no line was lost or passed another.
    out.resume();
    let resp = tokio::time::timeout(LONG, waiting).await.unwrap().unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(log.finish(None, LONG).await, 0);
    let ids: Vec<String> = out
        .lines()
        .iter()
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            record["request_id"].as_str().unwrap().to_string()
        })
        .collect();
    let expected: Vec<String> = (0..=answered).map(|n| format!("request-{n}")).collect();
    assert_eq!(ids, expected);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}
