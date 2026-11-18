//! Issue #709 on the shared tier: purge walks a page at a
//! time, refusals that keep a purge in force, and the bound
//! on group write-backs waiting for a write slot. Each test
//! names the part of the fix it pins.

use super::*;
use crate::s3store::Refused;

#[tokio::test]
async fn a_purge_request_lists_one_page_inline_whatever_the_bucket_holds() {
    // Issue #709: a purge request listed the whole bucket before
    // it answered, 31 pages for 31,000 objects (about a second),
    // and the walk it left kept the listing while it waited for
    // a turn. The request now lists one page, of the inline
    // budget, and the walk lists on from its continuation
    // token, a page at a time.
    use crate::s3store::PURGE_READS_AT_ONCE;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, 3000).await;
    *fake.walk.hold_prefix.lock().unwrap() = Some(TAIL_FILLER.into());
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    wait_until("the walk parks", || {
        fake.walk.held.load(Ordering::SeqCst) == PURGE_READS_AT_ONCE
    })
    .await;
    let walked = |fake: &crate::s3store::test_support::FakeS3| -> Vec<Option<usize>> {
        fake.walk
            .pages
            .lock()
            .unwrap()
            .iter()
            .filter(|(prefix, _)| prefix.is_empty())
            .map(|(_, max_keys)| *max_keys)
            .collect()
    };
    // The request's page, then the walk's first: the rest of
    // the bucket is not listed yet.
    assert_eq!(walked(&fake), vec![Some(SHARED_PURGE_INLINE_SCAN), None]);
    *fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&probe).await;
    assert_eq!(fake.ranged_gets.load(Ordering::SeqCst), 3000);
    assert_eq!(walked(&fake).len(), 4, "128, then 1000, 1000 and 872");
    assert!(bucket_tombstones(&fake)[0].ended_ms.is_some());
}

#[tokio::test]
async fn a_purge_whose_deletes_the_bucket_refuses_keeps_its_tombstone() {
    // Issue #709: a covered copy whose DELETE the bucket refused
    // (a 403, for want of s3:DeleteObject) counted as kept, so
    // the walk ended complete and marked its tombstone ended:
    // once the grace had passed, the copy served again on every
    // replica. The walk is now incomplete and the tombstone
    // stays in force; the refusal counts.
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, 10).await;
    let shared = probe.shared_store.clone().unwrap();
    let key = "40x24/example.com/stale.png#cfgtest";
    let member = CachedImage {
        groups: vec!["small".to_string()],
        ..tier_test_entry()
    };
    shared.put(key, &member).await;
    fake.fail.lock().unwrap().refuse_delete = true;
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    wait_for_purges_to_end(&probe).await;
    assert!(
        fake.objects
            .lock()
            .unwrap()
            .contains_key(&format!("{}.bin", crate::storage::key_stem(key)))
    );
    let tombstones = bucket_tombstones(&fake);
    assert_eq!(tombstones.len(), 1);
    assert_eq!(tombstones[0].ended_ms, None, "the purge stays in force");
    assert_eq!(shared.refused(Refused::Purge), 1);
    // Once the bucket allows it, the next purge removes the copy
    // and ends.
    fake.fail.lock().unwrap().refuse_delete = false;
    assert_eq!(purge_group_call(&base, "small").await, "purged 1\n");
    wait_for_purges_to_end(&probe).await;
    assert!(shared.get(key).await.is_none());
    assert!(bucket_tombstones(&fake).last().unwrap().ended_ms.is_some());
}

#[tokio::test]
async fn group_write_backs_past_the_waiting_cap_are_skipped_and_counted() {
    // Issue #709: group write-backs are deduplicated per copy,
    // but each distinct one parked a task of its own, holding its
    // entry, until a write slot freed, with no limit: while a
    // stalled bucket held every slot, they piled up with every
    // newly learned group. At most SHARED_GROUP_WRITES_WAITING
    // wait now; past them one is skipped and counted, as a
    // render's write-through is, and the disk tier is told the
    // bucket copy lacks its groups. A free slot is still taken
    // at once.
    let fake = crate::s3store::test_support::start_fake().await;
    fake.hold_puts.store(true, Ordering::SeqCst);
    let mut state = AppState {
        background: Background::with_group_waits(2, 3),
        ..tier_test_state(None, &fake.endpoint)
    };
    state.metrics_token = Some("metrics-token".to_string().into());
    let state = Arc::new(state);
    let base = serve_state(&state).await;
    let snapshot = state.purge_guard.snapshot();
    let grouped = CachedImage {
        groups: vec!["small".to_string()],
        ..tier_test_entry()
    };
    let key = |i: usize| format!("{i}/example.com/a.png");
    let send = |i: usize| {
        spawn_shared_put(
            &state,
            snapshot,
            &key(i),
            grouped.clone(),
            SharedWrite::Groups,
        )
    };
    // Two take the free slots at once.
    for i in 0..2 {
        assert_eq!(send(i), Sent::Spawned);
    }
    wait_until("both slots are held", || {
        fake.held_puts.load(Ordering::SeqCst) == 2
    })
    .await;
    assert_eq!(state.background.group_writes_waiting(), 0);
    // Three wait, one per copy, and the next is skipped.
    for i in 2..5 {
        assert_eq!(send(i), Sent::Spawned);
    }
    assert_eq!(state.background.group_writes_waiting(), 3);
    assert_eq!(send(5), Sent::Owed);
    assert_eq!(state.background.group_writes_skipped(), 1);
    // A write-back of a copy one already waiting carries is
    // still carried by it, not skipped.
    assert_eq!(send(2), Sent::Spawned);
    assert_eq!(state.background.group_writes_skipped(), 1);
    let metrics = scrape(&base, "metrics-token").await;
    assert!(
        metrics.contains("crabcrop_cache_shared_group_writes_waiting 3\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("crabcrop_cache_shared_group_writes_skipped_total 1\n"),
        "{metrics}"
    );
    // The bucket answers: every write-back that waited lands, and
    // none is left waiting.
    fake.hold_puts.store(false, Ordering::SeqCst);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
    assert_eq!(state.background.drain(deadline).await.writes, 0);
    assert_eq!(state.background.group_writes_waiting(), 0);
    assert_eq!(entry_objects(&fake), 5);
    assert!(
        state
            .shared_store
            .clone()
            .unwrap()
            .get(&key(5))
            .await
            .is_none()
    );
}

#[tokio::test]
async fn a_skipped_group_write_back_leaves_its_groups_owed_to_the_next_disk_hit() {
    // Issue #709: a group write-back skipped past the waiting cap
    // leaves the bucket copy without the group. The disk record
    // keeps the debt, so the next disk hit sends it again, and
    // the bucket copy learns the group a purge of it needs.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-owed-groups");
    let state = Arc::new(AppState {
        background: Background::with_group_waits(1, 0),
        ..tier_test_state(Some(&dir), &fake.endpoint)
    });
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    let key = "40x24/example.com/i.png";
    store.put(key, &tier_test_entry());
    shared.put(key, &tier_test_entry()).await;
    // The only write slot is held, and no write-back may wait.
    fake.hold_puts.store(true, Ordering::SeqCst);
    spawn_shared_put(
        &state,
        state.purge_guard.snapshot(),
        "other/example.com/o.png",
        tier_test_entry(),
        SharedWrite::Render,
    );
    wait_until("the slot is held", || {
        fake.held_puts.load(Ordering::SeqCst) == 1
    })
    .await;
    // A disk hit under `small`: the record learns it, and its
    // write-back is skipped.
    let groups = vec!["small".to_string()];
    let (_, detail) = lookup_tiers(&state, key, &groups).await.unwrap();
    assert_eq!(detail, DISK_HIT);
    assert_eq!(state.background.group_writes_skipped(), 1);
    // The bucket answers again. The next disk hit, under no new
    // group, pays the debt.
    fake.hold_puts.store(false, Ordering::SeqCst);
    wait_until("the slot is free", || {
        state.background.puts_in_flight() == 0
    })
    .await;
    state.cache.remove(key);
    let (_, detail) = lookup_tiers(&state, key, &[]).await.unwrap();
    assert_eq!(detail, DISK_HIT);
    shared_learns_group(&shared, key, "small").await;
}

#[tokio::test]
async fn refused_lookups_count_on_metrics() {
    // Issue #709: a lookup the bucket refused, the 403 of keys
    // it no longer honors, was a silent miss. It counts now,
    // by request, beside the tier's errors.
    let fake = crate::s3store::test_support::start_fake().await;
    let mut state = tier_test_state(None, &fake.endpoint);
    state.metrics_token = Some("metrics-token".to_string().into());
    let state = Arc::new(state);
    let base = serve_state(&state).await;
    let mut rotated = crate::s3store::test_support::test_credentials();
    rotated.access_key = "AKIDROTATED".to_string();
    *fake.credentials.lock().unwrap() = Some(rotated);
    assert!(
        lookup_tiers(&state, "40x24/example.com/i.png", &[])
            .await
            .is_none()
    );
    let metrics = scrape(&base, "metrics-token").await;
    assert!(
        metrics.contains("crabcrop_cache_shared_refused_total{request=\"lookup\"} 1\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("crabcrop_cache_tier_errors_total{tier=\"shared\"} 1\n"),
        "{metrics}"
    );
}

#[tokio::test]
async fn failed_upload_existence_checks_count_on_their_own_series() {
    // Issue #709: an upload's existence check, a HEAD to the
    // upload bucket, read as no object whenever it failed, with
    // nothing counted. It counts now, on the upload bucket's own
    // series: that bucket's store is not the shared tier, so its
    // failures move neither the shared tier's errors nor its
    // refusals, and it logs as `upload storage`.
    let fake = crate::s3store::test_support::start_fake().await;
    let upload = crate::upload::UploadStore::s3(
        "uploads".to_string(),
        String::new(),
        Some(fake.endpoint.clone()),
        "us-east-1".to_string(),
        Some(crate::s3store::test_support::test_aws()),
        1024,
        false,
    )
    .unwrap();
    let mut state = tier_test_state(None, &fake.endpoint);
    state.upload = Some(Arc::new(upload));
    state.metrics_token = Some("metrics-token".to_string().into());
    let state = Arc::new(state);
    let base = serve_state(&state).await;
    let upload = state.upload.clone().unwrap();
    // A name that is not there is an answer, not a failure.
    assert!(!upload.exists("cat.png").await);
    assert_eq!(upload.exists_checks_failed(), Some(0));
    // The bucket stops honoring the keys.
    let mut rotated = crate::s3store::test_support::test_credentials();
    rotated.access_key = "AKIDROTATED".to_string();
    *fake.credentials.lock().unwrap() = Some(rotated);
    assert!(!upload.exists("cat.png").await);
    let metrics = scrape(&base, "metrics-token").await;
    assert!(
        metrics.contains("crabcrop_upload_exists_checks_failed_total 1\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("crabcrop_cache_tier_errors_total{tier=\"shared\"} 0\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("crabcrop_cache_shared_refused_total{request=\"lookup\"} 0\n"),
        "{metrics}"
    );
    assert!(!metrics.contains("request=\"exists\""), "{metrics}");
}
