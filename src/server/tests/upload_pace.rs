//! The time an upload body may take once its signature has
//! verified (see `upload_body_due`), on tokio's paused clock:
//! a body's chunks arrive when the test schedules them, and
//! no test waits for one.

use super::*;
use std::time::Duration;

/// A body that sends `len` bytes `delay` after its previous
/// chunk was sent, one chunk per step, on the runtime's
/// clock. The schedule is the client's own: a chunk is due
/// when it is due, whenever the server reads it.
fn paced_body(steps: Vec<(Duration, usize)>) -> axum::body::Body {
    let start = tokio::time::Instant::now();
    let stream =
        futures_util::stream::unfold((steps.into_iter(), start), |(mut steps, sent)| async move {
            let (delay, len) = steps.next()?;
            let due = sent + delay;
            tokio::time::sleep_until(due).await;
            Some((
                Ok::<_, std::io::Error>(bytes::Bytes::from(vec![7u8; len])),
                (steps, due),
            ))
        });
    axum::body::Body::from_stream(stream)
}

fn unlimited_charge() -> crate::loader::SourceCharge {
    crate::loader::SourceBudget::unlimited().charge(SOURCE_BUDGET_WAIT)
}

#[tokio::test(start_paused = true)]
async fn an_upload_body_that_falls_behind_is_cut_off() {
    // A kilobyte a second never keeps up: once the grace
    // and the few milliseconds its bytes earn are spent, the
    // read stops with a timeout, about ten seconds in, long
    // before the body would end.
    let started = tokio::time::Instant::now();
    let body = paced_body(vec![(Duration::from_secs(1), 1024); 600]);
    let read = read_upload_body(body, 10 * 1024 * 1024, &unlimited_charge()).await;
    assert_eq!(read.unwrap_err(), LoadError::Timeout);
    let took = started.elapsed();
    assert!(
        took >= UPLOAD_BODY_GRACE && took < UPLOAD_BODY_GRACE + Duration::from_secs(1),
        "{took:?}"
    );
    // A body that sends nothing at all is cut off at the
    // grace itself.
    let started = tokio::time::Instant::now();
    let body = paced_body(vec![(Duration::from_secs(3600), 1024)]);
    let read = read_upload_body(body, 1024 * 1024, &unlimited_charge()).await;
    assert_eq!(read.unwrap_err(), LoadError::Timeout);
    assert_eq!(started.elapsed(), UPLOAD_BODY_GRACE);
}

#[tokio::test(start_paused = true)]
async fn an_upload_body_that_keeps_up_is_read_whole() {
    // A client that starts late, within the grace, and then
    // sends a little over 64 KiB a second takes as long as it
    // needs: here 9 seconds of silence and then 2.5 MiB over
    // 36 seconds.
    let mut steps = vec![(Duration::from_secs(9), 64 * 1024)];
    steps.extend(vec![(Duration::from_millis(900), 64 * 1024); 39]);
    let body = paced_body(steps);
    let read = read_upload_body(body, 10 * 1024 * 1024, &unlimited_charge()).await;
    assert_eq!(read.unwrap().len(), 40 * 64 * 1024);
    // So does a body that sends it all at once after most of
    // the grace.
    let body = paced_body(vec![(Duration::from_millis(9_900), 3 * 1024 * 1024)]);
    let read = read_upload_body(body, 10 * 1024 * 1024, &unlimited_charge()).await;
    assert_eq!(read.unwrap().len(), 3 * 1024 * 1024);
}

#[tokio::test(start_paused = true)]
async fn the_server_s_own_wait_for_budget_does_not_count_against_a_body() {
    // The budget is full for the first 4 seconds of the
    // read, so the server waits that long before it takes
    // the body's next chunk. Its last chunk then arrives 12.5
    // seconds in: past the grace, but within it once the
    // server's own wait is set aside.
    let budget = crate::loader::SourceBudget::new(512 * 1024);
    let hog = budget.charge(SOURCE_BUDGET_WAIT);
    hog.grow_to(512 * 1024).await.unwrap();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(4)).await;
        drop(hog);
    });
    let body = paced_body(vec![
        (Duration::ZERO, 1024),
        (Duration::from_millis(12_500), 1024),
    ]);
    let read = read_upload_body(body, 1024 * 1024, &budget.charge(SOURCE_BUDGET_WAIT)).await;
    assert_eq!(read.unwrap().len(), 2048);
}

#[tokio::test(start_paused = true)]
async fn an_upload_that_trickles_answers_408_and_stores_nothing() {
    let dir = upload_dir("trickle");
    std::fs::create_dir_all(&dir).unwrap();
    let dir = std::fs::canonicalize(&dir).unwrap();
    let state = Arc::new(AppState {
        upload: Some(Arc::new(crate::upload::UploadStore::local(
            dir.clone(),
            10 * 1024 * 1024,
            false,
        ))),
        ..test_state()
    });
    let name = "slow.png";
    let sig = crate::url::upload_signature(SECRET, name);
    let request = axum::http::Request::builder()
        .method("PUT")
        .uri(format!("/upload/{sig}/{name}"))
        .header(header::ACCEPT, "application/json")
        .body(paced_body(vec![(Duration::from_secs(2), 16); 10_000]))
        .unwrap();
    let response = upload_handle(State(state.clone()), request).await;
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("\"upload_too_slow\""), "{body}");
    assert!(!dir.join(name).exists());
    // The refusal is counted like every other upload answer.
    let metrics = state.metrics.render();
    assert!(metrics.contains("crabcrop_uploads_total 1"), "{metrics}");
}
