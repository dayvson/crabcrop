//! Issue #572 on the new disk tier: groups learned in the
//! index, claims, the bounded waits, and the cache-group rule
//! across them. Each test names the part of the fix it pins.

use super::disk_hang::{DiskSlotsHeld, STUCK_DISK_LIMIT, disk_only_state, get_within};
use super::*;
use crate::storage::{Bucket, OpKind};

/// Holds the disk operations of the given kinds on one store
/// until released or dropped, and records every operation it
/// sees, held or not. A hold also ends on its own after a
/// backstop, so a failed test cannot keep the runtime's drop
/// waiting for a blocking thread.
struct DiskOpGate {
    store: Arc<crate::storage::DiskStore>,
    shared: Arc<OpGateShared>,
}

struct OpGateShared {
    released: std::sync::Mutex<bool>,
    wake: std::sync::Condvar,
    seen: std::sync::Mutex<Vec<OpKind>>,
}

impl DiskOpGate {
    fn hold(
        store: &Arc<crate::storage::DiskStore>,
        kinds: &[OpKind],
    ) -> (Self, tokio::sync::mpsc::UnboundedReceiver<OpKind>) {
        let shared = Arc::new(OpGateShared {
            released: std::sync::Mutex::new(false),
            wake: std::sync::Condvar::new(),
            seen: std::sync::Mutex::new(Vec::new()),
        });
        let (held_tx, held_rx) = tokio::sync::mpsc::unbounded_channel();
        let hook = shared.clone();
        let kinds = kinds.to_vec();
        store.set_hold(Some(Arc::new(move |kind, _| {
            hook.seen.lock().unwrap().push(kind);
            if kinds.contains(&kind) {
                let _ = held_tx.send(kind);
                let released = hook.released.lock().unwrap();
                let _ = hook
                    .wake
                    .wait_timeout_while(released, std::time::Duration::from_secs(15), |r| !*r)
                    .unwrap();
            }
            Ok(())
        })));
        (
            DiskOpGate {
                store: store.clone(),
                shared,
            },
            held_rx,
        )
    }

    fn release(&self) {
        *self.shared.released.lock().unwrap() = true;
        self.shared.wake.notify_all();
    }

    /// How many operations of `kind` the gate has seen.
    fn seen(&self, kind: OpKind) -> usize {
        let seen = self.shared.seen.lock().unwrap();
        seen.iter().filter(|k| **k == kind).count()
    }
}

impl Drop for DiskOpGate {
    fn drop(&mut self) {
        self.store.set_hold(None);
        self.release();
    }
}

/// A tier-chain state with a disk tier at `dir` and no shared
/// tier.
fn disk_tier_state(dir: &std::path::Path) -> AppState {
    let mut state = tier_test_state(Some(dir), "http://127.0.0.1:9");
    state.shared_store = None;
    state
}

/// Presets with the same operation under each name: their
/// requests share one cache key.
fn presets_named(names: &[&str]) -> crate::url::Presets {
    names
        .iter()
        .map(|name| (name.to_string(), "30x20".parse().unwrap()))
        .collect()
}

fn in_groups(names: &[&str]) -> CachedImage {
    CachedImage {
        groups: names.iter().map(|g| g.to_string()).collect(),
        ..tier_test_entry()
    }
}

fn group_list(names: &[&str]) -> Vec<String> {
    names.iter().map(|g| g.to_string()).collect()
}

async fn disk_settles(store: &Arc<crate::storage::DiskStore>) {
    let left = store
        .settle(tokio::time::Instant::now() + std::time::Duration::from_secs(10))
        .await;
    assert_eq!(left, crate::storage::Backlog::default());
}

const KEY: &str = "40x24/example.com/i.png";

#[test]
fn the_stuck_disk_limit_covers_the_disk_bounds() {
    assert!(
        STUCK_DISK_LIMIT
            >= DISK_LOOKUP_BUDGET + DISK_WRITE_WAIT + std::time::Duration::from_secs(3)
    );
}

#[tokio::test]
async fn a_disk_budget_charges_only_the_time_spent_waiting() {
    let ms = std::time::Duration::from_millis;
    let mut budget = DiskBudget::with(ms(300));
    assert_eq!(budget.spend(tokio::time::sleep(ms(50))).await, Some(()));
    // Time outside the budget is not charged.
    tokio::time::sleep(ms(400)).await;
    assert!(!budget.spent());
    assert_eq!(budget.spend(tokio::time::sleep(ms(2000))).await, None);
    assert!(budget.spent());
    let started = std::time::Instant::now();
    assert_eq!(budget.spend(std::future::ready(1)).await, None);
    assert!(started.elapsed() < ms(50));
}

#[tokio::test]
async fn a_follower_after_a_long_render_still_reads_a_healthy_disk() {
    // The budget counts time spent waiting on the disk, not the
    // time since the request began: a request that waited for
    // a long render before its next check still reads a healthy
    // disk.
    let dir = crate::test_support::scratch_dir("srv-budget-wall");
    let state = disk_tier_state(&dir);
    state.store.as_ref().unwrap().put(KEY, &tier_test_entry());
    let mut budget = DiskBudget::with(std::time::Duration::from_millis(200));
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let (_, status) = lookup_tiers_within(&state, KEY, &[], &mut budget)
        .await
        .unwrap();
    assert_eq!(status, DISK_HIT);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_write_for_a_key_the_disk_holds_only_teaches_it_groups() {
    // Issue #572, race 1: a request whose disk lookup gave up
    // rendered and wrote its copy over the disk's, replacing
    // its groups. A key the index holds is never written
    // again now: its record learns the groups.
    let dir = crate::test_support::scratch_dir("srv-held-write");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let mut state = disk_only_state(&dir, 0);
    state.presets = presets_named(&["a", "c"]);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let resp = get_within(&base, &format!("preset:a/{src}"), "the first render").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let key = store.keys().pop().unwrap();
    let etag = store.get(&key).unwrap().etag;
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Read, OpKind::Write]);
    let resp = get_within(&base, &format!("preset:c/{src}"), "R1, whose read is stuck").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(gate.seen(OpKind::Write), 0, "the render rewrote the copy");
    drop(gate);
    disk_settles(&store).await;
    // The disk copy is the first render's, with both groups:
    // a purge of either reaches it, after a restart too.
    let reopened = test_store(&dir);
    let copy = reopened.get(&key).unwrap();
    assert_eq!((copy.etag, copy.groups), (etag, group_list(&["a", "c"])));
    let target = crate::cache::PurgeTarget::Group("a".into());
    assert_eq!(reopened.purge(&target).0, 1);
    drop(reopened);
    assert_eq!(purge_group_call(&base, "c").await, "purged 1\n");
    assert!(!store.contains(&key));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn an_abandoned_lookup_promotes_and_learns_nothing() {
    // Issue #572: a lookup task whose request went on without
    // it promotes nothing and teaches no tier anything.
    let dir = crate::test_support::scratch_dir("srv-abandoned");
    let state = disk_tier_state(&dir);
    let store = state.store.clone().unwrap();
    store.put(KEY, &tier_test_entry());
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Read]);
    let found = within(
        STUCK_DISK_LIMIT,
        "a lookup whose read is stuck",
        lookup_tiers(&state, KEY, &group_list(&["c"])),
    )
    .await;
    assert!(found.is_none());
    gate.release();
    wait_until("the lookup task ends", || {
        state.background.running().writes == 0
    })
    .await;
    drop(gate);
    assert!(state.cache.get(KEY).is_none(), "the late lookup promoted");
    assert_eq!(store.known_groups(KEY).unwrap(), Vec::<String>::new());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn groups_learned_on_memory_hits_during_a_stall_all_land() {
    // Issue #572, race 3, through the server: memory hits that
    // add groups while every disk write is stuck.
    let dir = crate::test_support::scratch_dir("srv-stall-learns");
    let state = disk_tier_state(&dir);
    let store = state.store.clone().unwrap();
    let keys: Vec<String> = (0..15)
        .map(|i| format!("{i}x24/example.com/i.png"))
        .collect();
    for key in &keys {
        store.put(key, &in_groups(&["a"]));
        state.cache.put(key.clone(), in_groups(&["a"]));
    }
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Write]);
    for group in ["c", "d"] {
        for key in &keys {
            let (_, status) = lookup_tiers(&state, key, &group_list(&[group]))
                .await
                .unwrap();
            assert_eq!(status, "crabcrop; hit");
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    drop(gate);
    disk_settles(&store).await;
    let reopened = test_store(&dir);
    for key in &keys {
        assert_eq!(
            reopened.known_groups(key).unwrap(),
            ["a", "c", "d"],
            "{key}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_fresh_write_that_lands_late_carries_a_group_learned_meanwhile() {
    // Issue #572, race 4: a write-through that stalled landed
    // with the groups it started with, missing one a memory
    // hit learned meanwhile.
    let dir = crate::test_support::scratch_dir("srv-late-write");
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let mut state = disk_only_state(&dir, 16 * 1024 * 1024);
    state.presets = presets_named(&["c", "d"]);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Write]);
    let resp = get_within(&base, &format!("preset:c/{src}"), "the render").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    let resp = get_within(&base, &format!("preset:d/{src}"), "the memory hit").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    drop(gate);
    wait_until("the held write lands", || !store.keys().is_empty()).await;
    disk_settles(&store).await;
    let key = store.keys().pop().unwrap();
    assert_eq!(store.known_groups(&key).unwrap(), ["c", "d"]);
    assert_eq!(test_store(&dir).get(&key).unwrap().groups, ["c", "d"]);
    assert_eq!(purge_group_call(&base, "d").await, "purged 2\n");
    let resp = get_within(&base, &format!("preset:c/{src}"), "after the purge").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn groups_learned_during_a_stall_survive_a_reopen() {
    let dir = crate::test_support::scratch_dir("srv-stall-reopen");
    let state = disk_tier_state(&dir);
    let store = state.store.clone().unwrap();
    store.put(KEY, &in_groups(&["a"]));
    state.cache.put(KEY.to_string(), in_groups(&["a"]));
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Write]);
    lookup_tiers(&state, KEY, &group_list(&["g"]))
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    drop(gate);
    disk_settles(&store).await;
    let reopened = test_store(&dir);
    let target = crate::cache::PurgeTarget::Group("g".into());
    assert_eq!(reopened.purge(&target).0, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_stuck_disk_and_a_stalled_bucket_queue_the_same_work_however_many_requests() {
    // Issue #572, race 5 with a bucket: with memory keeping
    // nothing and the disk stuck, every request is a bucket
    // hit, and each used to queue its own group write-back,
    // holding the image, until the first one landed.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-stuck-bucket");
    let mut state = tier_test_state(Some(&dir), &fake.endpoint);
    state.cache = Arc::new(Cache::new(0));
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    store.put(KEY, &in_groups(&["a"]));
    shared.put(KEY, &in_groups(&["a"])).await;
    fake.hold_puts.store(true, Ordering::SeqCst);
    let held = DiskSlotsHeld::hold(&store, 32).await;
    let mut work = Vec::new();
    for n in [4, 16] {
        let mut lookups = tokio::task::JoinSet::new();
        for _ in 0..n {
            let state = state.clone();
            lookups.spawn(async move {
                within(
                    STUCK_DISK_LIMIT,
                    "a lookup with the disk stuck",
                    lookup_tiers(&state, KEY, &["c".to_string()]),
                )
                .await
            });
        }
        while let Some(found) = lookups.join_next().await {
            let (_, status) = found.unwrap().unwrap();
            assert_eq!(status, "crabcrop; hit; detail=\"shared\"");
        }
        work.push((state.background.running().writes, store.backlog()));
    }
    assert_eq!(work[0], work[1], "the work grew with the requests");
    assert_eq!(work[0].0, 1);
    fake.hold_puts.store(false, Ordering::SeqCst);
    drop(held);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_request_dropped_before_its_disk_read_leaves_the_write_back_to_the_next_hit() {
    // A request learns its groups only once it is served: one
    // dropped while it waits for a disk slot leaves nothing
    // learned, so the next hit still sees the group as new and
    // writes it back to the bucket.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-drop-before-read");
    let state = tier_test_state(Some(&dir), &fake.endpoint);
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    store.put(KEY, &tier_test_entry());
    shared.put(KEY, &tier_test_entry()).await;
    let held = DiskSlotsHeld::hold(&store, 32).await;
    poll_lookup_once(&state, KEY, &group_list(&["thumb"]));
    drop(held);
    let (_, status) = lookup_tiers(&state, KEY, &group_list(&["thumb"]))
        .await
        .unwrap();
    assert_eq!(status, DISK_HIT);
    shared_learns_group(&shared, KEY, "thumb").await;
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn metrics_show_the_disk_gauges_while_every_slot_is_held() {
    let dir = crate::test_support::scratch_dir("srv-disk-gauges");
    let state = disk_only_state(&dir, 16 * 1024 * 1024);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let held = DiskSlotsHeld::hold(&store, 32).await;
    let metrics = within(
        STUCK_DISK_LIMIT,
        "/metrics",
        scrape(&base, TEST_METRICS_TOKEN),
    )
    .await;
    for line in [
        "\ncrabcrop_cache_disk_ops_in_flight 32\n",
        "\ncrabcrop_cache_disk_groups_behind 0\n",
        "\ncrabcrop_cache_disk_deletions_queued 0\n",
    ] {
        assert!(metrics.contains(line), "{line:?} missing from {metrics}");
    }
    drop(held);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_render_takes_its_own_bytes_and_keeps_the_groups_memory_held() {
    let mut state = tier_test_state(None, "http://127.0.0.1:9");
    state.shared_store = None;
    state.cache.put(KEY.to_string(), in_groups(&["a"]));
    let render = CachedImage {
        bytes: bytes::Bytes::from_static(b"a fresh render"),
        etag: crate::cache::etag_for_bytes(b"a fresh render"),
        written: Some(state.purge_guard.stamp()),
        ..in_groups(&["c"])
    };
    let snapshot = state.purge_guard.snapshot();
    populate_tiers(&state, snapshot, KEY, render.clone(), None).await;
    let held = state.cache.get(KEY).unwrap();
    assert!(crate::cache::same_copy(&held, &render));
    assert_eq!(held.groups, ["a", "c"]);
}

#[tokio::test]
async fn a_follower_does_not_wait_for_its_leaders_disk_write() {
    // A flight's followers read the render from memory once it
    // is there; the leader's disk write, held here, is
    // finished after they are woken.
    let dir = crate::test_support::scratch_dir("srv-handoff");
    let (src, hits) = spawn_slow_counting_server(test_png(), 300).await;
    let state = disk_only_state(&dir, 16 * 1024 * 1024);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Write]);
    let request = |base: String, remainder: String| async move {
        let resp = get_within(&base, &remainder, "a flight request").await;
        (std::time::Instant::now(), header(&resp, "cache-status"))
    };
    let remainder = format!("30x20/{src}");
    let leader = tokio::spawn(request(base.clone(), remainder.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let follower = tokio::spawn(request(base.clone(), remainder));
    let (leader_done, _) = leader.await.unwrap();
    let (follower_done, status) = follower.await.unwrap();
    assert!(follower_done < leader_done, "the follower waited");
    assert_eq!(status, "crabcrop; hit");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    drop(gate);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_follower_of_another_spelling_reads_the_leaders_disk_write_with_memory_off() {
    // With memory keeping nothing, a follower that cannot take
    // the leader's response reads the leader's disk write, so
    // the leader writes it before its followers wake.
    let dir = crate::test_support::scratch_dir("srv-handoff-off");
    let (src, hits) = spawn_slow_counting_server(test_png(), 300).await;
    let mut state = disk_only_state(&dir, 0);
    state.presets = presets_named(&["a", "b"]);
    let base = serve_state(&state).await;
    let leader = {
        let base = base.clone();
        let remainder = format!("preset:a/{src}");
        tokio::spawn(async move { get_within(&base, &remainder, "the leader").await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let follower = get_within(&base, &format!("preset:b/{src}"), "the follower").await;
    assert_eq!(header(&follower, "cache-status"), DISK_HIT);
    assert_eq!(leader.await.unwrap().status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_memory_hit_racing_a_group_purge_leaves_no_copy_in_the_group() {
    let mut state = tier_test_state(None, "http://127.0.0.1:9");
    state.shared_store = None;
    let hit = tier_test_entry();
    let groups = group_list(&["g"]);
    // A purge recorded after the hit's snapshot.
    state.cache.put(KEY.to_string(), hit.clone());
    let snapshot = state.purge_guard.snapshot();
    state.purge_guard.record_group("g");
    memory_hit(&state, snapshot, KEY, &groups, &hit);
    assert!(state.cache.get(KEY).is_none(), "a recorded purge");
    // A purge running when the hit took its snapshot, of a copy
    // written before it began.
    state.cache.put(KEY.to_string(), hit.clone());
    let _running = state
        .purge_guard
        .begin(crate::cache::PurgeTarget::Group("g".into()));
    let snapshot = state.purge_guard.snapshot();
    memory_hit(&state, snapshot, KEY, &groups, &hit);
    assert!(state.cache.get(KEY).is_none(), "a running purge");
}

#[tokio::test]
async fn a_render_whose_bucket_write_was_skipped_leaves_the_group_to_the_next_disk_hit() {
    // A render whose disk lookup gave up, whose bucket read
    // failed and whose bucket write found no slot teaches only
    // the disk record its group. The record owes it to the
    // bucket, and the next disk hit pays it.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-owed");
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let mut state = tier_test_state(Some(&dir), &fake.endpoint);
    state.cache = Arc::new(Cache::new(0));
    state.presets = presets_named(&["c"]);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    let resp = get_within(&base, &format!("30x20/{src}"), "the first render").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    writes_settle(&state).await;
    let (key, copy) = bucket_entry(&fake);
    assert!(copy.groups.is_empty());

    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Read]);
    fake.fail.lock().unwrap().get_object = true;
    let slots: Vec<_> = (0..crate::server::background::SHARED_PUTS_AT_ONCE)
        .map(|_| state.background.put_slot().unwrap())
        .collect();
    let resp = get_within(&base, &format!("preset:c/{src}"), "R1").await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    drop(slots);
    fake.fail.lock().unwrap().get_object = false;
    drop(gate);
    assert_eq!(store.backlog().owed, 1);

    let resp = get_within(&base, &format!("preset:c/{src}"), "R2").await;
    assert_eq!(header(&resp, "cache-status"), DISK_HIT);
    shared_learns_group(&shared, &key, "c").await;
    writes_settle(&state).await;
    group_purge_reaches_every_tier(&state, &base, &fake, (&src, &hits), "c").await;
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_bucket_stall_queues_no_write_back_per_rendered_key() {
    // A render's bucket write skipped for want of a slot (issue
    // #463) stays skipped: the disk hits that follow queue no
    // write-back for it.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-no-owed-render");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let mut state = tier_test_state(Some(&dir), &fake.endpoint);
    state.cache = Arc::new(Cache::new(0));
    let base = serve_state(&state).await;
    let slots: Vec<_> = (0..crate::server::background::SHARED_PUTS_AT_ONCE)
        .map(|_| state.background.put_slot().unwrap())
        .collect();
    for status in ["crabcrop; fwd=uri-miss", DISK_HIT] {
        for w in 10..18 {
            let resp = get_within(&base, &format!("{w}x21/{src}"), "a request").await;
            assert_eq!(header(&resp, "cache-status"), status);
        }
    }
    assert_eq!(state.background.running().writes, 0);
    assert_eq!(state.store.as_ref().unwrap().backlog().owed, 0);
    drop(slots);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_group_learned_under_a_live_purge_of_it_removes_the_disk_copy() {
    let dir = crate::test_support::scratch_dir("srv-live-purge-learn");
    let state = disk_tier_state(&dir);
    let store = state.store.clone().unwrap();
    store.put(KEY, &tier_test_entry());
    state.cache.put(KEY.to_string(), tier_test_entry());
    let _running = state
        .purge_guard
        .begin(crate::cache::PurgeTarget::Group("c".into()));
    let (_, status) = lookup_tiers(&state, KEY, &group_list(&["c"]))
        .await
        .unwrap();
    assert_eq!(status, "crabcrop; hit");
    assert!(!store.contains(KEY), "the disk copy outlives the purge");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_copy_served_during_a_purge_of_its_group_still_teaches_the_disk_copy() {
    let dir = crate::test_support::scratch_dir("srv-stale-teaches");
    let state = disk_tier_state(&dir);
    let store = state.store.clone().unwrap();
    store.put(KEY, &tier_test_entry());
    let snapshot = state.purge_guard.snapshot();
    state.purge_guard.record_group("c");
    let copy = in_groups(&["c"]);
    assert!(begin_disk_write(&state, snapshot, KEY, &copy, Bucket::Unknown).is_none());
    assert!(!store.contains(KEY), "the disk copy outlives the purge");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_group_only_memory_held_reaches_the_disk_and_the_bucket() {
    // Memory learned `d` while the disk held no record or
    // write of the key. A bucket hit that read the bucket
    // before then lands its copy on the disk: it takes the
    // groups memory holds, so a purge of `d` reaches it.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-memory-only-group");
    let state = tier_test_state(Some(&dir), &fake.endpoint);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    shared.put(KEY, &tier_test_entry()).await;
    fake.hold_object_gets.store(true, Ordering::SeqCst);
    let late = {
        let state = state.clone();
        tokio::spawn(async move { lookup_tiers(&state, KEY, &["e".to_string()]).await })
    };
    wait_until("the bucket read is held", || {
        fake.held_object_gets.load(Ordering::SeqCst) == 1
    })
    .await;
    state.cache.put(KEY.to_string(), in_groups(&["c"]));
    let (_, status) = lookup_tiers(&state, KEY, &group_list(&["d"]))
        .await
        .unwrap();
    assert_eq!(status, "crabcrop; hit");
    fake.hold_object_gets.store(false, Ordering::SeqCst);
    let (_, status) = within(STUCK_DISK_LIMIT, "the late bucket hit", late)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, "crabcrop; hit; detail=\"shared\"");
    writes_settle(&state).await;
    let disk = store.known_groups(KEY).unwrap();
    assert!(
        disk.contains(&"d".to_string()),
        "the disk copy holds {disk:?}"
    );
    purge_group_call(&base, "d").await;
    wait_for_purges_to_end(&state).await;
    assert!(lookup_tiers(&state, KEY, &[]).await.is_none());
    assert_eq!(entry_objects(&fake), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_purge_whose_disk_files_are_not_deleted_yet_leaves_its_tombstone_unended() {
    // A restart before a purge's disk files are deleted indexes
    // them again. The purge's bucket tombstone stays unended
    // until they are gone, so it keeps covering them.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-unended-tombstone");
    let state = tier_test_state(Some(&dir), &fake.endpoint);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    store.put(KEY, &in_groups(&["g"]));
    let (gate, _held) = DiskOpGate::hold(&store, &[OpKind::Unlink]);
    let answer = within(
        std::time::Duration::from_secs(3),
        "the purge",
        purge_group_call(&base, "g"),
    )
    .await;
    assert_eq!(answer, "purged 1\n");
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let tombstones = bucket_tombstones(&fake);
    assert_eq!(tombstones.len(), 1);
    assert_eq!(
        tombstones[0].ended_ms, None,
        "ended with the file still there"
    );
    assert_eq!(state.purge_guard.active_count(), 1);
    drop(gate);
    wait_until("the tombstone is marked ended", || {
        bucket_tombstones(&fake)[0].ended_ms.is_some()
    })
    .await;
    assert!(
        !dir.join(format!("{}.bin", crate::storage::key_stem(KEY)))
            .exists()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_slow_write_that_has_its_slot_is_not_a_stall() {
    let dir = crate::test_support::scratch_dir("srv-slow-write");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let state = disk_only_state(&dir, 16 * 1024 * 1024);
    let base = serve_state(&state).await;
    let store = state.store.clone().unwrap();
    store.set_hold(Some(Arc::new(|kind, _| {
        if kind == OpKind::Write {
            std::thread::sleep(std::time::Duration::from_millis(1500));
        }
        Ok(())
    })));
    let errors = store.errors();
    let resp = get_within(&base, &format!("30x20/{src}"), "a slow write").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(store.errors(), errors, "a slow write counted as a stall");
    store.set_hold(None);
    let held = DiskSlotsHeld::hold(&store, 32).await;
    let resp = get_within(&base, &format!("31x20/{src}"), "a write with no slot").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(store.errors(), errors + 1);
    drop(held);
    let _ = std::fs::remove_dir_all(&dir);
}
