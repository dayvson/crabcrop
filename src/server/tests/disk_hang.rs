//! A disk that stops answering (issue #572). These tests use
//! only what main already had (the read hook, `blocking`,
//! `lookup_tiers`, `running()` and HTTP), so this file alone
//! applies to a build without the fix, where each one fails
//! at [`STUCK_DISK_LIMIT`] naming the step that hung.

use super::*;

/// How long a request may take while the disk is stuck: the
/// 5 s of disk lookups and the 1 s disk write a request may
/// wait, with room for a loaded runner.
pub(super) const STUCK_DISK_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

/// A served state on a disk tier at `dir`, with no shared
/// tier, the test presets and a memory budget of `memory`.
pub(super) fn disk_only_state(dir: &std::path::Path, memory: usize) -> AppState {
    let mut state = tier_test_state(Some(dir), "http://127.0.0.1:9");
    state.shared_store = None;
    state.cache = Arc::new(Cache::new(memory));
    state.presets = preset_test_map();
    state.metrics_token = Some(TEST_METRICS_TOKEN.to_string());
    state
}

/// Holds every disk slot of a store, the way operations stuck
/// in the kernel hold them, until dropped. Each slot is held
/// by an operation that waits on a flag, so dropping the
/// guard lets them all go on every exit path. A hold also
/// ends on its own after a backstop, so a failed test cannot
/// keep the runtime's drop waiting for a blocking thread.
pub(super) struct DiskSlotsHeld {
    release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

impl DiskSlotsHeld {
    /// Takes `n` slots of `store` and returns once each is held.
    pub(super) async fn hold(store: &Arc<crate::storage::DiskStore>, n: usize) -> Self {
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let (started_tx, mut started) = tokio::sync::mpsc::unbounded_channel();
        for _ in 0..n {
            let store = store.clone();
            let release = release.clone();
            let started_tx = started_tx.clone();
            tokio::spawn(async move {
                let _ = store
                    .blocking(move |_| {
                        let _ = started_tx.send(());
                        let (flag, wake) = &*release;
                        let released = flag.lock().unwrap();
                        let _ = wake
                            .wait_timeout_while(
                                released,
                                std::time::Duration::from_secs(60),
                                |released| !*released,
                            )
                            .unwrap();
                    })
                    .await;
            });
        }
        for _ in 0..n {
            within(STUCK_DISK_LIMIT, "a disk slot to be held", started.recv())
                .await
                .unwrap();
        }
        DiskSlotsHeld { release }
    }
}

impl Drop for DiskSlotsHeld {
    fn drop(&mut self) {
        let (flag, wake) = &*self.release;
        *flag.lock().unwrap() = true;
        wake.notify_all();
    }
}

/// GET `remainder` within [`STUCK_DISK_LIMIT`], naming `what`.
pub(super) async fn get_within(base: &str, remainder: &str, what: &str) -> reqwest::Response {
    within(STUCK_DISK_LIMIT, what, get_signed(base, remainder)).await
}

/// The keys the hang tests seed: more than the store has
/// slots, so a stuck disk holds every slot with some left
/// waiting.
fn seeded_remainders(src: &str) -> Vec<String> {
    (10..50).map(|w| format!("{w}x21/{src}")).collect()
}

/// Serve `state`, render every seeded key and `30x20/{src}`
/// into its memory and disk, then serve a second state over
/// the same store with its memory cold, holding
/// `30x20/{src}` in memory. Returns that second state and
/// its base URL.
async fn seed_then_go_cold(state: &AppState, src: &str) -> (AppState, String) {
    let base = serve_state(state).await;
    let mut remainders = seeded_remainders(src);
    remainders.push(format!("30x20/{src}"));
    for remainder in &remainders {
        let resp = get_within(&base, remainder, "a seeding render").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
    assert_eq!(state.store.as_ref().unwrap().stats().0, remainders.len());
    let mut cold = state.clone();
    cold.cache = Arc::new(Cache::new(16 * 1024 * 1024));
    let cold_base = serve_state(&cold).await;
    let resp = get_within(&cold_base, &format!("30x20/{src}"), "the warming hit").await;
    assert_eq!(
        header(&resp, "cache-status"),
        "crabcrop; hit; detail=\"disk\""
    );
    (cold, cold_base)
}

/// Send, all at once, a request for every seeded key, a fresh
/// render, and a memory hit that adds the `small` group, and
/// check each answers 200 within [`STUCK_DISK_LIMIT`], and
/// within what its kind may wait on a stuck disk: a key the
/// disk holds 5 s for its lookups, a fresh render 1 s for its
/// write, a memory hit nothing. Each limit leaves 2 s for the
/// render and the load of a test run, 1 s for the memory hit.
async fn every_kind_of_request_answers(base: &str, src: &str) {
    let secs = std::time::Duration::from_secs;
    let mut requests = tokio::task::JoinSet::new();
    let mut kinds: Vec<(String, std::time::Duration)> = seeded_remainders(src)
        .into_iter()
        .map(|remainder| (remainder, secs(5) + secs(2)))
        .collect();
    kinds.push((format!("60x21/{src}"), secs(1) + secs(2)));
    kinds.push((format!("preset:small/{src}"), secs(1)));
    for (remainder, limit) in kinds {
        let base = base.to_string();
        requests.spawn(async move {
            let started = std::time::Instant::now();
            let resp = get_within(&base, &remainder, &format!("GET {remainder}")).await;
            (remainder, limit, resp.status(), started.elapsed())
        });
    }
    while let Some(answer) = requests.join_next().await {
        let (remainder, limit, status, took) = answer.unwrap();
        assert_eq!(status, StatusCode::OK, "{remainder}");
        assert!(
            took <= limit,
            "GET {remainder} took {took:?}, over {limit:?}"
        );
    }
}

#[tokio::test]
async fn requests_answer_while_every_disk_read_is_stuck() {
    // Issue #572: with every disk read stuck, a lookup held
    // its slot forever, and once all 32 were held, every
    // lookup, write-through and group merge waited for one,
    // so every request that read the cache hung. Fails on a
    // build without the fix at the limit below.
    let dir = crate::test_support::scratch_dir("srv-stuck-reads");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let state = disk_only_state(&dir, 16 * 1024 * 1024);
    let (cold, base) = seed_then_go_cold(&state, &src).await;
    let store = cold.store.clone().unwrap();
    let errors = store.errors();
    let (gate, _held) = DiskReadGate::hold(&store);
    every_kind_of_request_answers(&base, &src).await;
    // At most one lookup task per slot is left behind.
    assert!(cold.background.running().writes <= 32);
    assert!(store.errors() > errors, "the stalls were not counted");
    drop(gate);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_shared_hit_answers_while_the_disk_read_is_stuck() {
    // Issue #572, with a bucket: a lookup whose disk read is
    // stuck goes on to the shared tier.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-stuck-shared");
    let state = tier_test_state(Some(&dir), &fake.endpoint);
    let store = state.store.clone().unwrap();
    let key = "40x24/example.com/i.png";
    store.put(key, &tier_test_entry());
    state
        .shared_store
        .as_ref()
        .unwrap()
        .put(key, &tier_test_entry())
        .await;
    let (gate, _held) = DiskReadGate::hold(&store);
    let (_, status) = within(
        STUCK_DISK_LIMIT,
        "a lookup whose disk read is stuck",
        lookup_tiers(&state, key, &["thumb".to_string()]),
    )
    .await
    .unwrap();
    assert_eq!(status, "crabcrop; hit; detail=\"shared\"");
    drop(gate);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn requests_answer_while_every_disk_slot_is_held() {
    // Issue #572: the same requests with every disk slot held
    // and no read stuck. A lookup used to spawn its task and
    // then wait for a slot with no limit; now no task waits
    // for a slot, and a memory hit does not wait on the disk.
    let dir = crate::test_support::scratch_dir("srv-stuck-slots");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let state = disk_only_state(&dir, 16 * 1024 * 1024);
    let (cold, base) = seed_then_go_cold(&state, &src).await;
    let store = cold.store.clone().unwrap();
    let held = DiskSlotsHeld::hold(&store, 32).await;
    // The memory hit that adds a group answers at once.
    let started = std::time::Instant::now();
    let resp = get_within(&base, &format!("preset:small/{src}"), "the memory hit").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "the memory hit waited {:?}",
        started.elapsed()
    );
    let requests = {
        let base = base.clone();
        let src = src.clone();
        tokio::spawn(async move { every_kind_of_request_answers(&base, &src).await })
    };
    // While they wait, no lookup task waits for a slot.
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
    assert_eq!(cold.background.running().writes, 0);
    within(STUCK_DISK_LIMIT, "every request", requests)
        .await
        .unwrap();
    drop(held);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn purges_and_an_upload_overwrite_answer_while_the_disk_is_stuck() {
    // Issue #572: a purge, a group purge and the purge behind
    // an upload overwrite each waited for a disk slot with no
    // limit. Now they answer, with exact counts, and the
    // entries they purged never serve again.
    let dir = crate::test_support::scratch_dir("srv-stuck-purge");
    let uploads = crate::test_support::scratch_dir("srv-stuck-purge-up");
    std::fs::create_dir_all(&uploads).unwrap();
    let uploads_root = std::fs::canonicalize(&uploads).unwrap();
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let mut state = disk_only_state(&dir, 16 * 1024 * 1024);
    state.loader = Loader::with_defaults()
        .unwrap()
        .with_sources(crate::loader::SourceConfig {
            file_root: Some(uploads_root.clone()),
            ..Default::default()
        });
    state.upload = Some(Arc::new(crate::upload::UploadStore::local(
        uploads_root,
        10 * 1024 * 1024,
        true,
    )));
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let upload = format!(
        "{base}/upload/{}/cat.png",
        crate::url::upload_signature(SECRET, "cat.png")
    );
    let resp = client().put(&upload).body(test_png()).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let (by_key, by_group, by_source) = (
        format!("40x24/{src}"),
        format!("preset:small/{src}"),
        "50x30/file:///cat.png".to_string(),
    );
    for remainder in [&by_key, &by_group, &by_source] {
        let resp = get_within(&base, remainder, "a seeding render").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
    assert_eq!(store.stats().0, 3);
    let rendered = hits.load(Ordering::SeqCst);

    let held = DiskSlotsHeld::hold(&store, 32).await;
    let limit = std::time::Duration::from_secs(3);
    let sig = crate::url::purge_signature(SECRET, &by_key);
    let resp = within(
        limit,
        "the purge",
        client().post(format!("{base}/purge/{sig}/{by_key}")).send(),
    )
    .await
    .unwrap();
    assert_eq!(resp.text().await.unwrap(), "purged 2\n");
    assert_eq!(store.stats().0, 2, "the purge left the disk index alone");
    let answer = within(limit, "the group purge", purge_group_call(&base, "small")).await;
    assert_eq!(answer, "purged 2\n");
    assert_eq!(store.stats().0, 1);
    let other = {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            90,
            50,
            image::Rgb([0, 0, 255]),
        ));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    };
    let resp = within(
        limit,
        "the upload overwrite",
        client().put(&upload).body(other).send(),
    )
    .await
    .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(store.stats().0, 0);
    // Before the disk answers, the purged entries render.
    for remainder in [&by_key, &by_group] {
        let resp = get_within(&base, remainder, "a purged entry").await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    }
    assert_eq!(hits.load(Ordering::SeqCst), rendered + 2);
    drop(held);
    // Once it answers, the purged files go, and a store
    // reopened on the directory indexes none of them.
    let entry_files = || {
        std::fs::read_dir(&dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "bin")
            })
            .count()
    };
    // The two renders above write their own entries.
    within(STUCK_DISK_LIMIT, "the purged files to go", async {
        while entry_files() != store.stats().0 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    })
    .await;
    let reopened = test_store(&dir);
    assert_eq!(reopened.stats().0, store.stats().0);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&uploads);
}

#[tokio::test]
async fn a_stuck_disk_queues_no_work_per_request() {
    // Issue #572, race 5: with memory keeping nothing, every
    // request for a key the disk holds reaches the disk. With
    // every slot held, each one that added a group queued its
    // own merge task. Now the work stays the same however
    // many requests come.
    let dir = crate::test_support::scratch_dir("srv-stuck-per-request");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let state = disk_only_state(&dir, 0);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let resp = get_within(&base, &format!("30x20/{src}"), "the seeding render").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(store.stats().0, 1);
    let held = DiskSlotsHeld::hold(&store, 32).await;
    let mut writes = Vec::new();
    for n in [4, 16] {
        let mut requests = tokio::task::JoinSet::new();
        for _ in 0..n {
            let base = base.clone();
            let remainder = format!("preset:small/{src}");
            requests.spawn(async move {
                get_within(&base, &remainder, "a request for a key the disk holds")
                    .await
                    .status()
            });
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        writes.push(state.background.running().writes);
        while let Some(status) = requests.join_next().await {
            assert_eq!(status.unwrap(), StatusCode::OK);
        }
    }
    assert_eq!(writes[0], writes[1], "the work grew with the requests");
    drop(held);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_lookup_that_answers_late_leaves_no_copy_a_group_purge_misses() {
    // Issue #572, race 2, the probe: a lookup whose disk read
    // answers after its request went on to the bucket put its
    // own view back into memory and the bucket, taking back a
    // group a memory hit had added meanwhile. A purge of that
    // group then missed the copies. Hangs at R1 on a build
    // without the fix.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-late-lookup");
    let state = tier_test_state(Some(&dir), &fake.endpoint);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    let key = "40x24/example.com/i.png";
    let in_a = CachedImage {
        groups: vec!["a".to_string()],
        ..tier_test_entry()
    };
    store.put(key, &in_a);
    shared.put(key, &tier_test_entry()).await;
    let (gate, _held) = DiskReadGate::hold(&store);
    let (_, status) = within(
        STUCK_DISK_LIMIT,
        "R1, whose disk read is stuck",
        lookup_tiers(&state, key, &["c".to_string()]),
    )
    .await
    .unwrap();
    assert_eq!(status, "crabcrop; hit; detail=\"shared\"");
    shared_learns_group(&shared, key, "c").await;
    let (_, status) = within(
        STUCK_DISK_LIMIT,
        "R2",
        lookup_tiers(&state, key, &["d".to_string()]),
    )
    .await
    .unwrap();
    assert_eq!(status, "crabcrop; hit");
    // The stuck read answers.
    gate.release();
    writes_settle(&state).await;
    shared_learns_group(&shared, key, "d").await;
    purge_group_call(&base, "d").await;
    wait_for_purges_to_end(&state).await;
    assert!(
        lookup_tiers(&state, key, &[]).await.is_none(),
        "a copy served under d outlived its purge"
    );
    assert_eq!(entry_objects(&fake), 0);
    drop(gate);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn two_disk_hits_at_once_keep_both_groups_in_memory() {
    // Issue #572, race 2, the two-hit case: two disk hits for
    // one key, each with its own group, each built its set
    // from its own read and put it into memory and the disk
    // file, so one group was lost.
    let dir = crate::test_support::scratch_dir("srv-two-hits");
    let mut state = tier_test_state(Some(&dir), "http://127.0.0.1:9");
    state.shared_store = None;
    let store = state.store.clone().unwrap();
    let key = "40x24/example.com/i.png";
    store.put(key, &tier_test_entry());
    let (gate, mut held) = DiskReadGate::hold(&store);
    let lookups: Vec<_> = ["c", "d"]
        .into_iter()
        .map(|group| {
            let state = state.clone();
            tokio::spawn(async move {
                within(
                    STUCK_DISK_LIMIT,
                    "a disk hit",
                    lookup_tiers(&state, key, &[group.to_string()]),
                )
                .await
            })
        })
        .collect();
    for _ in 0..2 {
        within(STUCK_DISK_LIMIT, "both reads to be held", held.recv())
            .await
            .unwrap();
    }
    gate.release();
    for lookup in lookups {
        let (_, status) = lookup.await.unwrap().unwrap();
        assert_eq!(status, "crabcrop; hit; detail=\"disk\"");
    }
    drop(gate);
    let memory = state.cache.get(key).unwrap().groups;
    assert!(
        memory.contains(&"c".to_string()) && memory.contains(&"d".to_string()),
        "memory holds {memory:?}"
    );
    let disk = store.get(key).unwrap().groups;
    assert!(
        disk.contains(&"c".to_string()) && disk.contains(&"d".to_string()),
        "the disk holds {disk:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
