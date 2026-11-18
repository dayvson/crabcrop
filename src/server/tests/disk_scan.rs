//! The disk tier's boot scan on a thread of its own (issue
//! #708): the server answers while the scan is stuck, and what
//! the scan finds afterwards is judged by what was done
//! meanwhile.

use super::disk_hang::{STUCK_DISK_LIMIT, disk_only_state, get_within};
use super::*;
use crate::storage::{OpKind, ScanPhase};

/// Holds every read the boot scan of `store` makes, on its own
/// thread only, until dropped, and reports each read it holds.
/// A hold also ends on its own after a backstop, so a failed
/// test cannot keep the scan thread waiting.
struct ScanReadsHeld {
    released: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

impl ScanReadsHeld {
    fn hold(
        store: &Arc<crate::storage::DiskStore>,
    ) -> (
        Self,
        tokio::sync::mpsc::UnboundedReceiver<std::path::PathBuf>,
    ) {
        let released = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let (held_tx, held) = tokio::sync::mpsc::unbounded_channel();
        let gate = released.clone();
        store.set_hold(Some(Arc::new(move |kind, path| {
            if kind == OpKind::Read
                && std::thread::current().name() == Some(crate::storage::SCAN_THREAD)
            {
                let _ = held_tx.send(path.to_path_buf());
                let (flag, wake) = &*gate;
                let done = flag.lock().unwrap();
                let _ = wake
                    .wait_timeout_while(done, std::time::Duration::from_secs(30), |done| !*done)
                    .unwrap();
            }
            Ok(())
        })));
        (ScanReadsHeld { released }, held)
    }
}

impl Drop for ScanReadsHeld {
    fn drop(&mut self) {
        let (flag, wake) = &*self.released;
        *flag.lock().unwrap() = true;
        wake.notify_all();
    }
}

#[tokio::test]
async fn the_server_answers_while_the_boot_scan_is_stuck_and_a_purge_sent_meanwhile_holds() {
    // Issue #708: the boot scan ran before the server listened,
    // so a disk that hung at boot left every request, /health
    // included, unanswered. With the scan stuck on a thread of
    // its own, the server answers: /health, a lookup of an
    // entry the scan has not indexed, which renders, and a
    // purge. Once the scan goes on, the entry the purge covers
    // does not come back.
    let dir = crate::test_support::scratch_dir("srv-scan-stuck");
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let purged = format!("30x20/{src}");
    let rendered = format!("40x20/{src}");
    // A previous run renders both into its disk tier.
    {
        let state = Arc::new(disk_only_state(&dir, 16 * 1024 * 1024));
        let base = serve_state(&state).await;
        for remainder in [&purged, &rendered] {
            let resp = get_within(&base, remainder, "a seeding render").await;
            assert_eq!(resp.status(), StatusCode::OK);
        }
        assert_eq!(state.store.as_ref().unwrap().stats().0, 2);
    }
    let seeded = hits.load(Ordering::SeqCst);
    let store = Arc::new(crate::storage::DiskStore::unscanned(
        &dir,
        16 * 1024 * 1024,
        crate::url::entry_key(b"disk-key-test"),
    ));
    let (held, mut reads) = ScanReadsHeld::hold(&store);
    store.begin_scan();
    within(STUCK_DISK_LIMIT, "the scan's first read", reads.recv())
        .await
        .unwrap();
    let state = Arc::new(AppState {
        store: Some(store.clone()),
        presets: preset_test_map(),
        metrics_token: Some(TEST_METRICS_TOKEN.to_string().into()),
        ..test_state()
    });
    let base = serve_state(&state).await;
    let health = within(
        STUCK_DISK_LIMIT,
        "/health while the scan is stuck",
        client().get(format!("{base}/health")).send(),
    )
    .await
    .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    let resp = get_within(&base, &rendered, "a lookup while the scan is stuck").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(hits.load(Ordering::SeqCst), seeded + 1);
    let sig = crate::url::purge_signature(SECRET, &purged);
    let resp = within(
        STUCK_DISK_LIMIT,
        "a purge while the scan is stuck",
        client().post(format!("{base}/purge/{sig}/{purged}")).send(),
    )
    .await
    .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(held);
    let scanned = store.clone();
    let report = tokio::task::spawn_blocking(move || {
        scanned.wait_scan(
            std::time::Instant::now() + STUCK_DISK_LIMIT,
            ScanPhase::ended,
        )
    })
    .await
    .unwrap();
    assert_eq!(report.phase, ScanPhase::Done);
    // The copy rendered meanwhile is indexed, the purged one is
    // not, and its file goes.
    let keys = store.keys();
    assert_eq!(keys.len(), 1, "{keys:?}");
    assert!(keys[0].starts_with("40x20/"), "{keys:?}");
    let left = store
        .settle(tokio::time::Instant::now() + STUCK_DISK_LIMIT)
        .await;
    assert_eq!(left, crate::storage::Backlog::default());
    let files = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".bin")
        })
        .count();
    assert_eq!(files, 1);
    let resp = get_within(&base, &rendered, "a hit after the scan").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = get_within(&base, &purged, "the purged key after the scan").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(hits.load(Ordering::SeqCst), seeded + 2);
}

#[tokio::test]
async fn a_tombstone_read_while_the_boot_scan_runs_keeps_the_copy_it_covers_from_serving() {
    // Issue #708: a purge sent to another replica reaches this
    // one as a bucket tombstone, which can be read while the
    // scan runs. The scan indexes the copy the tombstone covers
    // like any other, and its first lookup is a miss that
    // deletes it, as for a copy indexed before the tombstone
    // came: the purged copy never serves.
    let dir = crate::test_support::scratch_dir("srv-scan-tombstone");
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let remainder = format!("30x20/{src}");
    let key = {
        let state = Arc::new(disk_only_state(&dir, 16 * 1024 * 1024));
        let base = serve_state(&state).await;
        let resp = get_within(&base, &remainder, "a seeding render").await;
        assert_eq!(resp.status(), StatusCode::OK);
        state.store.as_ref().unwrap().keys().pop().unwrap()
    };
    let seeded = hits.load(Ordering::SeqCst);
    let store = Arc::new(crate::storage::DiskStore::unscanned(
        &dir,
        16 * 1024 * 1024,
        crate::url::entry_key(b"disk-key-test"),
    ));
    let (held, mut reads) = ScanReadsHeld::hold(&store);
    store.begin_scan();
    within(STUCK_DISK_LIMIT, "the scan's first read", reads.recv())
        .await
        .unwrap();
    let state = Arc::new(AppState {
        store: Some(store.clone()),
        presets: preset_test_map(),
        metrics_token: Some(TEST_METRICS_TOKEN.to_string().into()),
        ..test_state()
    });
    let base = serve_state(&state).await;
    // Another replica's purge of the key, still running.
    let base_key = key.split('#').next().unwrap().to_string();
    state
        .purge_guard
        .replace_shared(std::collections::HashMap::from([(
            "purges/elsewhere".to_string(),
            crate::cache::Tombstone {
                target: crate::cache::PurgeTarget::Key(base_key),
                cutoff: crate::cache::PurgeCutoff {
                    start_ms: crate::cache::now_ms(),
                    writer: 7,
                },
                ended_ms: None,
            },
        )]));
    drop(held);
    let scanned = store.clone();
    let report = tokio::task::spawn_blocking(move || {
        scanned.wait_scan(
            std::time::Instant::now() + STUCK_DISK_LIMIT,
            ScanPhase::ended,
        )
    })
    .await
    .unwrap();
    assert_eq!(report.phase, ScanPhase::Done);
    assert!(store.contains(&key), "the scan indexes the copy");
    let resp = get_within(&base, &remainder, "a lookup of the covered copy").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(hits.load(Ordering::SeqCst), seeded + 1);
}
