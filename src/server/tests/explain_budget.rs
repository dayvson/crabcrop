//! `/explain`'s source probe and the source buffer budget
//! (issue #403). The probe charges the head it reads to the
//! budget as the bytes arrive, as a cache miss charges its
//! source, and the head stays charged until the probe is done
//! with it, its wait for the probe gate included. So a probe
//! past the budget waits for budget and then sheds with the
//! miss's own `503`, and `crabcrop_source_buffer_bytes`
//! counts the heads that probes hold.

use super::*;
use crate::loader::SourceBudget;
use std::time::Duration;

/// A served app on `loader` and `probe_gate`, with the test
/// metrics token.
async fn spawn_budgeted(loader: Loader, probe_gate: Arc<Gate>) -> String {
    spawn_state(AppState {
        loader,
        probe_gate,
        metrics_token: Some(TEST_METRICS_TOKEN.to_string().into()),
        ..test_state()
    })
    .await
}

/// The `crabcrop_source_buffer_bytes` gauge.
async fn held_bytes(base: &str) -> usize {
    metrics_text(base)
        .await
        .lines()
        .find_map(|line| line.strip_prefix("crabcrop_source_buffer_bytes "))
        .and_then(|value| value.parse().ok())
        .expect("the gauge is exported")
}

/// Wait at most `limit` for the gauge to read `bytes`.
async fn gauge_reads(base: &str, bytes: usize, limit: Duration) {
    let deadline = Instant::now() + limit;
    loop {
        let held = held_bytes(base).await;
        if held == bytes {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the gauge reads {held} bytes after {limit:?}, not {bytes}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// `GET` `path` from `base` as JSON: the status, the
/// `Retry-After` header, and the body.
async fn get_json(base: &str, path: &str) -> (StatusCode, Option<String>, serde_json::Value) {
    let resp = client()
        .get(format!("{base}/{path}"))
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let retry_after = resp
        .headers()
        .get(header::RETRY_AFTER)
        .map(|value| value.to_str().unwrap().to_string());
    let body = resp.text().await.unwrap();
    let json = serde_json::from_str(&body).unwrap_or_else(|_| panic!("not JSON: {body}"));
    (status, retry_after, json)
}

/// An origin that answers every request with a declared
/// length of `declared` bytes and sends the first `sent` of
/// them, all zeros, then holds the connection until the test
/// notifies it, and only then sends the rest.
async fn spawn_held_origin(declared: usize, sent: usize) -> (String, Arc<tokio::sync::Notify>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let release = Arc::new(tokio::sync::Notify::new());
    let released = release.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let released = released.clone();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(&vec![0u8; sent]).await;
                released.notified().await;
                let _ = socket.write_all(&vec![0u8; declared - sent]).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/held.png"), release)
}

#[tokio::test]
async fn an_explain_probe_past_the_source_buffer_budget_sheds_as_a_miss_does() {
    // Something else holds the whole budget, so the probe's
    // head cannot be charged: the probe waits for budget for
    // the miss's five seconds, then sheds with the miss's own
    // answer, as the image request for the same URL does.
    // Charged nothing, the probe answered at once, whatever
    // the budget held.
    const BUDGET: usize = 1 << 20;
    let budget = SourceBudget::new(BUDGET);
    let loader = Loader::with_defaults()
        .unwrap()
        .with_source_budget(budget.clone());
    let base = spawn_budgeted(loader, Arc::new(Gate::new(2))).await;
    let remainder = format!("100x50/{}", spawn_image_server().await);
    let signature = sign(&remainder);
    let explain = format!("explain/{signature}/{remainder}");
    let image = format!("{signature}/{remainder}");
    let hog = budget.charge(Duration::ZERO);
    hog.grow_to(BUDGET).await.unwrap();
    let start = Instant::now();
    let (probe, miss) = tokio::join!(get_json(&base, &explain), get_json(&base, &image));
    let waited = start.elapsed();
    assert_eq!(probe.0, StatusCode::SERVICE_UNAVAILABLE, "{}", probe.2);
    assert_eq!(probe.1.as_deref(), Some("1"));
    assert_eq!(probe.2["error"]["code"], "server_busy");
    assert_eq!(
        probe.2["error"]["message"],
        "server busy: too many source images held in memory"
    );
    assert_eq!(probe, miss, "the probe and the miss shed alike");
    assert!(
        waited >= Duration::from_millis(4500),
        "shed after {waited:?}, before the wait bound"
    );
    assert_eq!(budget.in_use(), BUDGET, "a shed probe holds nothing");
    // With the budget free again, the same probe is served,
    // and holds nothing once it has answered.
    drop(hog);
    let (status, _, report) = get_json(&base, &explain).await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["source"]["kind"], "png");
    assert_eq!(budget.in_use(), 0, "the probe's charge came back");
}

#[tokio::test]
async fn the_gauge_counts_the_head_a_slow_origin_keeps_a_probe_holding() {
    // The origin declares 200,000 bytes, sends the first
    // 100,000 at once, and then nothing until the test lets
    // it go on. The head identifies no source, so the probe
    // reads on and keeps what it has: once its first windows
    // are over, it holds exactly the bytes it received, and
    // the gauge counts them while it waits. Charged nothing,
    // the probe held them and the gauge read zero.
    const DECLARED: usize = 200_000;
    const SENT: usize = 100_000;
    let budget = SourceBudget::new(64 << 20);
    let loader = Loader::with_defaults()
        .unwrap()
        .with_source_budget(budget.clone());
    let base = spawn_budgeted(loader, Arc::new(Gate::new(2))).await;
    let (src, release) = spawn_held_origin(DECLARED, SENT).await;
    let remainder = format!("100x50/{src}");
    let path = format!("explain/{}/{remainder}", sign(&remainder));
    let probe = {
        let base = base.clone();
        tokio::spawn(async move { get_json(&base, &path).await })
    };
    gauge_reads(&base, SENT, Duration::from_secs(5)).await;
    assert!(!probe.is_finished(), "the probe is still reading");
    // The rest arrives: the probe reads to the end of the
    // body, reports it, and gives its charge back.
    release.notify_one();
    let (status, _, report) = probe.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["source"]["kind"], "unknown");
    assert_eq!(report["source"]["bytes"], DECLARED);
    assert_eq!(held_bytes(&base).await, 0, "the probe's charge came back");
}

#[tokio::test]
async fn a_heavy_probe_keeps_its_head_charged_while_it_waits_for_the_probe_gate() {
    // A HEIC is sized on a blocking thread under the probe
    // gate, whose one slot the test holds: the probe has read
    // the whole head and waits for the slot with it, so the
    // head stays charged until the probe has run, from an
    // origin and from a file alike. Charged nothing, the
    // waiting heads were never counted.
    let heic = include_bytes!("../../../tests/fixtures/orient6.heic");
    let dir = crate::test_support::scratch_dir("explain-budget-heic");
    let root = std::fs::canonicalize(&dir).unwrap();
    std::fs::write(root.join("o.heic"), heic).unwrap();
    let budget = SourceBudget::new(64 << 20);
    let loader = Loader::with_defaults()
        .unwrap()
        .with_sources(crate::loader::SourceConfig {
            file_root: Some(root),
            ..Default::default()
        })
        .with_source_budget(budget.clone());
    let gate = Arc::new(Gate::new(1));
    let base = spawn_budgeted(loader, gate.clone()).await;
    let origin = spawn_bytes_server(heic.to_vec()).await;
    for src in [origin.as_str(), "file:///o.heic"] {
        let slot = gate.clone().acquire_owned().await.unwrap();
        let remainder = format!("64x64/{src}");
        let path = format!("explain/{}/{remainder}", sign(&remainder));
        let probe = {
            let base = base.clone();
            tokio::spawn(async move { get_json(&base, &path).await })
        };
        gauge_reads(&base, heic.len(), Duration::from_secs(4)).await;
        assert!(!probe.is_finished(), "{src}: the probe waits for the slot");
        drop(slot);
        let (status, _, report) = probe.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{src}: {report}");
        assert_eq!(report["source"]["kind"], "heic", "{src}");
        assert_eq!(held_bytes(&base).await, 0, "{src}: the charge came back");
    }
    drop(dir);
}
