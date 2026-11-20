use std::sync::Arc;

use super::{AppState, background};
use crate::cache::CachedImage;

#[cfg(doc)]
use super::{SHARED_GROUP_WRITES_WAITING, SHARED_PUTS_AT_ONCE};

/// Whether `entry` must not stand in any tier: a purge
/// recorded after `snapshot` covers it, so the write that
/// carries the snapshot began before that purge (issue
/// #350), or a live tombstone covers the copy by its stamp,
/// as when it was rendered before another replica's purge
/// began (issue #461).
fn write_is_stale(state: &AppState, snapshot: u64, key: &str, entry: &CachedImage) -> bool {
    state
        .purge_guard
        .write_covered(snapshot, key, &entry.groups, entry.written)
}

/// Store `entry` in the memory tier under the purge
/// guard (issue #350). The write is skipped when a purge
/// covering it already landed; a purge that lands while
/// the write is in flight is reconciled right after: the
/// entry just written is removed again, so the purge
/// wins however the two interleave. The removal matches
/// on the entry's ETag and stamp, so an entry a fresher
/// write has already replaced under the same key is left
/// alone. A copy memory held keeps its groups in the new
/// one (issue #572); returns those the slower tiers still
/// have to learn.
pub(in crate::server) fn put_memory_tier(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
) -> Vec<String> {
    if write_is_stale(state, snapshot, key, entry) {
        return Vec::new();
    }
    let took = state.cache.put(key.to_string(), entry.clone());
    let stored = crate::cache::union_groups(&entry.groups, &took);
    if state
        .purge_guard
        .write_covered(snapshot, key, &stored, entry.written)
        && state
            .cache
            .get(key)
            .is_some_and(|held| crate::cache::same_copy(&held, entry))
    {
        state.cache.remove(key);
    }
    took
}

/// Promote a copy a slower tier served into memory under
/// the purge guard (issue #350). A copy memory already
/// holds keeps its bytes and gains the groups (issue
/// #572), so a promotion that lands late never takes back a
/// group a hit taught memory meanwhile. Returns the groups
/// memory held that `entry` lacked, which the slower tiers
/// still have to learn.
fn promote_memory_tier(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
) -> Vec<String> {
    if write_is_stale(state, snapshot, key, entry) {
        return Vec::new();
    }
    match state.cache.promote(key, entry) {
        crate::cache::Promoted::Inserted => {
            if write_is_stale(state, snapshot, key, entry)
                && state
                    .cache
                    .get(key)
                    .is_some_and(|held| crate::cache::same_copy(&held, entry))
            {
                state.cache.remove(key);
            }
            Vec::new()
        }
        crate::cache::Promoted::Merged {
            groups,
            took,
            grew,
            written,
        } => {
            // A held copy that grows into a group a live purge
            // covers leaves memory, as on a memory hit.
            let covered = if grew {
                state
                    .purge_guard
                    .write_covered(snapshot, key, &groups, written)
            } else {
                state.purge_guard.purge_covers(snapshot, key, &groups)
            };
            if covered {
                state.cache.remove(key);
            }
            took
        }
        crate::cache::Promoted::Skipped => Vec::new(),
    }
}

/// What a learn tells the disk tier about the bucket copy,
/// given how its write went (issue #572): only a write that
/// was spawned, or is queued, carries the groups, and one
/// skipped past the waiting cap leaves the bucket copy without
/// them (issue #709).
fn bucket_for(sent: Sent, groups: &[String]) -> crate::storage::Bucket<'_> {
    match sent {
        Sent::Spawned => crate::storage::Bucket::Carries(groups),
        Sent::Skipped => crate::storage::Bucket::Skipped,
        Sent::Owed => crate::storage::Bucket::Lacks,
        Sent::Nothing => crate::storage::Bucket::Unknown,
    }
}

/// Teach the disk tier that a copy of `key` was served
/// under `groups` (issue #572): the index records them at
/// once, with no file I/O, and the drain writes them into
/// the file. When the record grows into a group a purge
/// covers, the disk copy goes: the reconcile a disk merge
/// ran on main, with the record's own stamp and id.
fn learn_disk_groups(
    state: &AppState,
    snapshot: u64,
    key: &str,
    groups: &[String],
    bucket: crate::storage::Bucket<'_>,
) {
    let Some(store) = &state.store else {
        return;
    };
    if let crate::storage::Learned::Indexed {
        grew: true,
        groups,
        written,
        id,
    } = store.learn_groups(key, groups, bucket)
    {
        if state
            .purge_guard
            .write_covered(snapshot, key, &groups, written)
        {
            store.forget_if(key, id);
        }
        store.kick();
    }
}

/// Teach the disk and shared tiers the groups memory held
/// for `key` that `entry`, the copy a caller just wrote to
/// them, lacked (issue #572): memory served the entry under
/// them, so every copy has to carry them.
fn teach_slower_tiers(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
    took: Vec<String>,
) {
    if took.is_empty() {
        return;
    }
    let copy = CachedImage {
        groups: crate::cache::union_groups(&entry.groups, &took),
        ..entry.clone()
    };
    let sent = spawn_shared_put(state, snapshot, key, copy.clone(), SharedWrite::Groups);
    learn_disk_groups(
        state,
        snapshot,
        key,
        &copy.groups,
        bucket_for(sent, &copy.groups),
    );
}

/// The groups this process knows `key` has been served
/// under (issue #572): the disk record's, or those of a disk
/// write in flight, and the memory copy's. A copy that serves
/// instead of those takes them, so no copy written below
/// lacks one.
fn known_groups(state: &AppState, key: &str) -> Vec<String> {
    let disk = state
        .store
        .as_ref()
        .and_then(|store| store.known_groups(key))
        .unwrap_or_default();
    match state.cache.groups(key) {
        Some(memory) => crate::cache::union_groups(&disk, &memory),
        None => disk,
    }
}

/// Whether `extra` holds a group `existing` lacks.
fn adds_groups(existing: &[String], extra: &[String]) -> bool {
    extra.iter().any(|g| !existing.contains(g))
}

/// Start a disk write of `entry` under the purge guard (issue
/// #572). A key the disk holds is never written again: its
/// record learns the groups, and when that grows it into a
/// group a purge covers, the disk copy goes, as main's merge
/// and reconcile did. A stale copy is not written. Returns
/// the claim of a write to finish with
/// [`finish_disk_write`]. Runs before the copy enters memory,
/// so a memory hit that learns a group meanwhile reaches the
/// write.
pub(in crate::server) fn begin_disk_write(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
    bucket: crate::storage::Bucket<'_>,
) -> Option<crate::storage::WriteClaim> {
    let store = state.store.as_ref()?;
    let may_claim = !write_is_stale(state, snapshot, key, entry);
    match store.begin_write(key, &entry.groups, bucket, may_claim) {
        crate::storage::WritePlan::Claimed(claim) => Some(claim),
        crate::storage::WritePlan::Held(crate::storage::Learned::Indexed {
            grew: true,
            groups,
            written,
            id,
        }) => {
            if state
                .purge_guard
                .write_covered(snapshot, key, &groups, written)
            {
                store.forget_if(key, id);
            }
            store.kick();
            None
        }
        _ => None,
    }
}

/// Finish a disk write begun by [`begin_disk_write`] (issue
/// #572). The request waits at most [`DISK_WRITE_WAIT`] for
/// it. A write that has no disk slot by then is dropped, with
/// its claim, and counted as a stall; one that has started
/// runs to its end on its thread. The write lands with every
/// group learned meanwhile, and the reconcile after it checks
/// those under the write's own snapshot: a purge that lands
/// while the write is in flight wins. With `owes`, the bucket
/// copy lacks some of the entry's groups, because its group
/// write-back was skipped (issue #709): the new record owes
/// them, and its first hit sends them.
pub(in crate::server) async fn finish_disk_write(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
    claim: crate::storage::WriteClaim,
    owes: bool,
) {
    let Some(store) = state.store.clone() else {
        return;
    };
    let started = tokio::time::Instant::now();
    let permit = match tokio::time::timeout(DISK_WRITE_WAIT, store.permit()).await {
        Ok(Some(permit)) => permit,
        Ok(None) => return,
        Err(_) => {
            store.note_stall(crate::storage::Stall::Write);
            return;
        }
    };
    let guard = state.purge_guard.clone();
    let key = key.to_string();
    let entry = entry.clone();
    let write = store.blocking_with(permit, move |store| {
        if guard.write_covered(snapshot, &key, &entry.groups, entry.written) {
            return;
        }
        let Some(landed) = store.land(claim, &entry) else {
            return;
        };
        if owes {
            store.owe_shared(&key, landed.id);
        }
        if guard.write_covered(snapshot, &key, &landed.groups, entry.written) {
            // Queued, not deleted inline: a failed delete is
            // retried, and the purge holds its tombstones for it
            // (see `DiskStore::forget_if`).
            store.forget_if(&key, landed.id);
        }
    });
    let _ = tokio::time::timeout(DISK_WRITE_WAIT.saturating_sub(started.elapsed()), write).await;
}

/// A render's disk write that its flight leader finishes
/// after the followers are woken (issue #572).
pub(in crate::server) struct PendingDiskWrite {
    pub(in crate::server) snapshot: u64,
    pub(in crate::server) key: String,
    pub(in crate::server) entry: CachedImage,
    pub(in crate::server) claim: crate::storage::WriteClaim,
}

/// Where [`populate_tiers`] leaves a flight leader's disk
/// write for the leader to finish once its guard is dropped.
pub(in crate::server) type DiskHandoff = Arc<std::sync::Mutex<Option<PendingDiskWrite>>>;

/// Why a shared-tier write is made, which decides what it
/// does when every write slot is taken (issue #463).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::server) enum SharedWrite {
    /// A fresh render's write-through, on a miss. Skipped
    /// when every slot is taken: the bucket only misses a
    /// copy that the next replica to miss renders again.
    Render,
    /// A tier hit's write-back of the request's cache groups
    /// onto the stored copy (issue #350). Never skipped: the
    /// local merge has already recorded the groups, so no
    /// later hit sends them again, and a bucket copy that
    /// never learns a group outlives a purge of it on every
    /// replica. It waits for a slot in its own task instead.
    Groups,
}

/// Write `entry` through to the shared tier in the
/// background under the purge guard (issue #350): the
/// same skip-and-reconcile discipline as the local
/// tiers, applied inside the spawned task, because the
/// purge it must lose to can land at any point during
/// the network write. The reconcile deletes only the
/// object it read back (issue #446), so a fresher render
/// that replaced it meanwhile stays.
///
/// At most [`SHARED_PUTS_AT_ONCE`] writes are in flight
/// (issue #463), each holding its entry until the bucket
/// answers. With every slot taken, a stalled bucket's
/// backlog, a [`SharedWrite::Render`] is skipped and counted
/// on /metrics: the local tiers hold the entry, so the cost
/// is a render on the next replica that misses. A
/// [`SharedWrite::Groups`] write-back waits for a slot in its
/// own task. One for the same copy, snapshot and groups as
/// one already queued is not spawned again (issue #572): the
/// queued one carries them, so a stall cannot queue one per
/// request. And at most
/// [`SHARED_GROUP_WRITES_WAITING`] wait (issue
/// #709): past them a write-back is skipped and counted on
/// /metrics, and comes back [`Sent::Owed`], so the disk tier
/// records that the bucket copy lacks the groups and its next
/// hit sends them again.
pub(in crate::server) fn spawn_shared_put(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: CachedImage,
    write: SharedWrite,
) -> Sent {
    let Some(shared) = &state.shared_store else {
        return Sent::Nothing;
    };
    let guard = state.purge_guard.clone();
    // A write a purge already covers would be dropped in the
    // task anyway: checked before it takes a slot, so it is
    // not counted as a skip.
    if guard.write_covered(snapshot, key, &entry.groups, entry.written) {
        return Sent::Nothing;
    }
    let (slot, queued) = match write {
        SharedWrite::Render => match state.background.put_slot() {
            Some(slot) => (Some(slot), None),
            None => return Sent::Skipped,
        },
        SharedWrite::Groups => match state.background.queue_groups_write(key, snapshot, &entry) {
            background::GroupsQueued::Queued(queued, slot) => (slot, Some(queued)),
            background::GroupsQueued::Carried => return Sent::Spawned,
            background::GroupsQueued::Skipped => return Sent::Owed,
        },
    };
    let background = state.background.clone();
    let shared = shared.clone();
    let key = key.to_string();
    // The handle is dropped: the write is detached, and
    // tracked for the shutdown drain.
    let _detached = state.background.spawn(async move {
        let mut queued = queued;
        let _slot = match slot {
            Some(slot) => slot,
            None => match background.wait_put_slot().await {
                Some(slot) => slot,
                None => return,
            },
        };
        if let Some(queued) = &mut queued {
            queued.got_slot();
        }
        // Checked again with the slot held: a purge can begin
        // while the write waits for one.
        if guard.write_covered(snapshot, &key, &entry.groups, entry.written) {
            return;
        }
        shared.put(&key, &entry).await;
        if guard.write_covered(snapshot, &key, &entry.groups, entry.written)
            && let Some((held, etag)) = shared.get_tagged(&key).await
            && crate::cache::same_copy(&held, &entry)
        {
            shared.remove_if_match(&key, etag.as_deref()).await;
        }
    });
    Sent::Spawned
}

/// How [`spawn_shared_put`] went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::server) enum Sent {
    /// A write was spawned, or one already queued carries the
    /// copy.
    Spawned,
    /// A render's write-through found every slot taken (issue
    /// #463).
    Skipped,
    /// A group write-back found as many waiting as may (issue
    /// #709): the bucket copy lacks its groups.
    Owed,
    /// Nothing to write: no shared tier, or a purge covers the
    /// copy.
    Nothing,
}

/// How long one request waits for the disk tier's lookups,
/// in all (issue #572): the waits for a disk slot and for the
/// reads, across every tier check the request makes. A
/// lookup that runs out goes on to the shared tier or a
/// render.
pub(in crate::server) const DISK_LOOKUP_BUDGET: std::time::Duration =
    std::time::Duration::from_secs(5);

/// How long a request waits for its own disk write, once
/// (issue #572). A write with no disk slot by then is
/// skipped.
pub(in crate::server) const DISK_WRITE_WAIT: std::time::Duration =
    std::time::Duration::from_secs(1);

/// What is left of a request's [`DISK_LOOKUP_BUDGET`]. Only
/// time spent waiting on the disk is charged: a request that
/// waited for a long render before its next check still
/// reads a healthy disk.
#[derive(Debug)]
pub(in crate::server) struct DiskBudget {
    left: std::time::Duration,
}

impl DiskBudget {
    pub(in crate::server) fn new() -> Self {
        DiskBudget {
            left: DISK_LOOKUP_BUDGET,
        }
    }

    #[cfg(test)]
    pub(in crate::server) fn with(left: std::time::Duration) -> Self {
        DiskBudget { left }
    }

    pub(in crate::server) fn spent(&self) -> bool {
        self.left.is_zero()
    }

    /// Wait for `wait` within what is left, and charge the
    /// time it took. `None`, at once once spent, when it ran
    /// out.
    pub(in crate::server) async fn spend<F: std::future::Future>(
        &mut self,
        wait: F,
    ) -> Option<F::Output> {
        if self.spent() {
            return None;
        }
        let started = tokio::time::Instant::now();
        let out = tokio::time::timeout(self.left, wait).await.ok();
        self.left = match out {
            Some(_) => self.left.saturating_sub(started.elapsed()),
            None => std::time::Duration::ZERO,
        };
        out
    }
}

/// Who acts on a disk lookup's answer (issue #572): the
/// request, if it is still waiting, or nobody. A lookup task
/// whose request went on without it promotes nothing and
/// writes nothing back.
#[derive(Debug, Default)]
struct LookupAnswer(std::sync::atomic::AtomicU8);

impl LookupAnswer {
    const WAITING: u8 = 0;
    const TAKEN: u8 = 1;
    const ABANDONED: u8 = 2;

    fn swap_from_waiting(&self, to: u8) -> bool {
        self.0
            .compare_exchange(
                Self::WAITING,
                to,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
    }

    /// The lookup task takes its answer. False when the
    /// request has gone on.
    fn take(&self) -> bool {
        self.swap_from_waiting(Self::TAKEN)
    }

    /// The request goes on without the answer. False when the
    /// task took it first: it only has memory work left.
    fn abandon(&self) -> bool {
        self.swap_from_waiting(Self::ABANDONED)
    }
}

/// The `Cache-Status` of a disk hit.
pub(in crate::server) const DISK_HIT: &str = "crabcrop; hit; detail=\"disk\"";

/// [`lookup_tiers_within`] with a whole disk budget, as the
/// tier tests drive one lookup.
#[cfg(test)]
pub(in crate::server) async fn lookup_tiers(
    state: &Arc<AppState>,
    cache_key: &str,
    groups: &[String],
) -> Option<(CachedImage, &'static str)> {
    lookup_tiers_within(state, cache_key, groups, &mut DiskBudget::new()).await
}

/// A memory hit's group learn (audit R2-15, issues #350,
/// #572): the copy memory holds gains the request's groups,
/// and when its set grows, the bucket copy and the disk
/// record learn the whole set. None of it waits on the disk:
/// the record learns in the index, and the drain writes the
/// file. When the grown set falls under a purge, the copy
/// leaves memory too, as the slower tiers' copies do.
pub(in crate::server) fn memory_hit(
    state: &AppState,
    snapshot: u64,
    key: &str,
    groups: &[String],
    hit: &CachedImage,
) {
    if groups.is_empty() {
        return;
    }
    let held = match state.cache.merge_groups(key, groups) {
        Some(held) => held,
        None => {
            // Memory may have dropped the copy since it was read:
            // the request is still served under its groups.
            if !adds_groups(&hit.groups, groups)
                || state
                    .cache
                    .groups(key)
                    .is_some_and(|memory| !adds_groups(&memory, groups))
            {
                return;
            }
            CachedImage {
                groups: crate::cache::union_groups(&hit.groups, groups),
                ..hit.clone()
            }
        }
    };
    let sent = spawn_shared_put(state, snapshot, key, held.clone(), SharedWrite::Groups);
    learn_disk_groups(
        state,
        snapshot,
        key,
        &held.groups,
        bucket_for(sent, &held.groups),
    );
    if state
        .purge_guard
        .write_covered(snapshot, key, &held.groups, held.written)
        && state
            .cache
            .get(key)
            .is_some_and(|now| crate::cache::same_copy(&now, &held))
    {
        state.cache.remove(key);
    }
}

/// The tier lookup chain for one key: memory, then disk
/// (a hit promotes into memory), then the shared tier (a
/// hit promotes into both local tiers). On any hit, the
/// request's cache groups are unioned into the stored
/// entry (audit R2-15): provenance accumulates, so an
/// entry first stored through a group-free spelling is
/// still removed by a later group purge once any grouped
/// spelling of the same URL has been served. The union
/// reaches every tier that holds the entry, not just the
/// one that served the hit (issue #350): membership that
/// stayed in memory alone would be lost at restart,
/// leaving the persistent copies invisible to the group
/// purge. Every write this chain makes (promotions,
/// merge write-backs) runs under the purge guard, under
/// the generation the lookup started with. A disk or
/// shared copy is a miss, and is deleted on sight, when a
/// live tombstone covers it: a purge, begun here or read
/// from the shared bucket, that is still running or within
/// its grace, and that began after the copy was written
/// (issues #400, #446, #461). The purge may not have
/// reached such a copy yet, and serving or promoting it
/// would outlive the purge; a copy rendered after the
/// purge began serves normally. The memory tier needs no
/// such check: a purge empties this process's memory
/// before any slower tier, and every way back in is a
/// promotion through this chain or a guarded write.
///
/// The disk tier is waited for within `budget` (issue
/// #572): a lookup that finds no disk slot, or whose read
/// does not answer, before the budget runs out goes on to
/// the shared tier, and its task promotes nothing. A
/// request learns groups only when it is served a copy, so
/// one dropped while it waits owes no tier anything.
/// Returns the image and the `Cache-Status` value for
/// the tier that served it.
pub(in crate::server) async fn lookup_tiers_within(
    state: &Arc<AppState>,
    cache_key: &str,
    groups: &[String],
    budget: &mut DiskBudget,
) -> Option<(CachedImage, &'static str)> {
    let snapshot = state.purge_guard.snapshot();
    if let Some(hit) = state.cache.get(cache_key) {
        memory_hit(state, snapshot, cache_key, groups, &hit);
        return Some((hit, "crabcrop; hit"));
    }
    if let Some(store) = &state.store
        && store.contains(cache_key)
        && !budget.spent()
    {
        // The slot is taken before the task starts, so no task
        // is left waiting for one, and at most one per slot
        // can be stuck in the kernel.
        match budget.spend(store.permit()).await {
            None => store.note_stall(crate::storage::Stall::LookupSlot),
            Some(None) => {}
            Some(Some(permit)) => {
                // The disk hit and everything it owes the other
                // tiers run in a spawned task (issue #347):
                // dropping this request's JoinHandle detaches
                // the task instead of cancelling it, so a
                // client that disconnects while the disk
                // answers still has its hit's groups learned
                // (issue #350). Tracked like every detached
                // task (issue #463).
                let answer = Arc::new(LookupAnswer::default());
                let mut task = state.background.spawn(lookup_disk_tier(
                    state.clone(),
                    snapshot,
                    cache_key.to_string(),
                    groups.to_vec(),
                    permit,
                    answer.clone(),
                ));
                match budget.spend(&mut task).await {
                    Some(Ok(Some(image))) => return Some((image, DISK_HIT)),
                    Some(_) => {}
                    None if answer.abandon() => store.note_stall(crate::storage::Stall::LookupRead),
                    // The task took its answer first: only memory
                    // work is left in it.
                    None => {
                        if let Ok(Some(image)) = task.await {
                            return Some((image, DISK_HIT));
                        }
                    }
                }
            }
        }
    }
    if let Some(shared) = &state.shared_store
        && let Some((mut image, etag)) = shared.get_tagged(cache_key).await
    {
        // Checked after the read: a walk can delete this
        // object and end while its bytes are in flight, and
        // an ended purge's tombstone still covers the copy.
        // The delete is conditional on the copy read, so a
        // fresh render that replaced it meanwhile stays.
        if state
            .purge_guard
            .running_covers(cache_key, &image.groups, image.written)
        {
            shared.remove_if_match(cache_key, etag.as_deref()).await;
            return None;
        }
        // The copy takes the request's groups and every group
        // this process knows for the key (issue #572): it
        // serves instead of the copies that were served under
        // them. When a purge covers one of those known groups,
        // the write-back would be dropped and the copy would
        // not learn the request's groups either, so it takes
        // the request's and the known ones no purge covers.
        let mut wanted = crate::cache::union_groups(groups, &known_groups(state, cache_key));
        let covered = |extra: &[String]| {
            state.purge_guard.write_covered(
                snapshot,
                cache_key,
                &crate::cache::union_groups(&image.groups, extra),
                image.written,
            )
        };
        if covered(&wanted) && !covered(groups) {
            let kept: Vec<String> = wanted
                .iter()
                .filter(|g| {
                    !covered(&crate::cache::union_groups(
                        groups,
                        std::slice::from_ref(*g),
                    ))
                })
                .cloned()
                .collect();
            wanted = crate::cache::union_groups(groups, &kept);
        }
        let sent = if adds_groups(&image.groups, &wanted) {
            image.groups = crate::cache::union_groups(&image.groups, &wanted);
            // Off the response path, like the write-through
            // put, and skipped only past the waiting cap
            // (issue #709), which the disk write below records
            // as a debt.
            spawn_shared_put(
                state,
                snapshot,
                cache_key,
                image.clone(),
                SharedWrite::Groups,
            )
        } else {
            // The bucket copy already carries them all.
            Sent::Spawned
        };
        // The disk write is claimed before the copy enters
        // memory, so a memory hit that learns a group meanwhile
        // reaches the write (issue #572).
        let disk = begin_disk_write(
            state,
            snapshot,
            cache_key,
            &image,
            bucket_for(sent, &image.groups),
        );
        let took = promote_memory_tier(state, snapshot, cache_key, &image);
        teach_slower_tiers(state, snapshot, cache_key, &image, took);
        if let Some(claim) = disk {
            finish_disk_write(
                state,
                snapshot,
                cache_key,
                &image,
                claim,
                sent == Sent::Owed,
            )
            .await;
        }
        return Some((image, "crabcrop; hit; detail=\"shared\""));
    }
    None
}

/// The disk tier's half of [`lookup_tiers_within`], run as its
/// own task under the disk slot its request took, so a
/// dropped request cannot strand it mid-chain: read the
/// entry, and if the request is still waiting for it, learn
/// the request's groups in the index under the purge guard,
/// then pay what the hit owes the other tiers: the group
/// write-back to the shared tier and the promotion into
/// memory, in that order. A request that went on without
/// the answer (issue #572) leaves the task nothing to do.
/// Returns the image on a hit.
async fn lookup_disk_tier(
    state: Arc<AppState>,
    snapshot: u64,
    key: String,
    groups: Vec<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
    answer: Arc<LookupAnswer>,
) -> Option<CachedImage> {
    let store = state.store.clone()?;
    let guard = state.purge_guard.clone();
    let read_key = key.clone();
    let (mut image, id) = store
        .blocking_with(permit, move |store| {
            let (image, id) = store.get_with_id(&read_key)?;
            if guard.running_covers(&read_key, &image.groups, image.written) {
                // A live tombstone covers this copy (issues
                // #400, #461): drop it rather than serve and
                // promote it. Its deletion is queued, retried
                // until done, and holds the tombstones of the
                // purge that covers it (issue #572).
                store.forget_if(&read_key, id);
                return None;
            }
            Some((image, id))
        })
        .await
        .flatten()?;
    if !answer.take() {
        return None;
    }
    let hit = store.learn_hit(&key, id, &groups)?;
    image.groups = hit.groups;
    if (hit.grew || hit.owed)
        // The record grew, or a skipped write left the bucket
        // copy without some of its groups: the shared tier's
        // copy learns them (issue #350). A write-back skipped
        // past the waiting cap leaves the debt with the record
        // again, for the next hit (issue #709).
        && spawn_shared_put(&state, snapshot, &key, image.clone(), SharedWrite::Groups) == Sent::Owed
    {
        store.owe_shared(&key, id);
    }
    if hit.grew
        && state
            .purge_guard
            .write_covered(snapshot, &key, &image.groups, image.written)
    {
        store.forget_if(&key, id);
    }
    let took = promote_memory_tier(&state, snapshot, &key, &image);
    teach_slower_tiers(&state, snapshot, &key, &image, took);
    store.kick();
    Some(image)
}

/// Store a fresh render in every tier under the purge
/// guard (issue #350), at the generation the render
/// started under: a purge that landed while this render
/// was running keeps the entry out of every tier, while
/// the response still serves the render itself. The shared
/// tier's write stays off the response path.
///
/// The stored copy takes every group this process knows for
/// the key (issue #572): it serves instead of the copies
/// served under them. The response's own headers still name
/// the request's groups. A key the disk holds is not written
/// again; its record learns the groups. With `handoff`, a
/// flight leader's disk write is left there for the leader
/// to finish after its followers are woken, but only when
/// memory holds this copy for them to read.
pub(in crate::server) async fn populate_tiers(
    state: &AppState,
    snapshot: u64,
    cache_key: &str,
    mut entry: CachedImage,
    handoff: Option<&DiskHandoff>,
) {
    let known = known_groups(state, cache_key);
    if adds_groups(&entry.groups, &known) {
        entry.groups = crate::cache::union_groups(&entry.groups, &known);
    }
    // The shared write is spawned before any wait (issue
    // #347): it is off the response path, and the order means
    // a request dropped during the disk write cannot strand
    // it.
    let sent = spawn_shared_put(
        state,
        snapshot,
        cache_key,
        entry.clone(),
        SharedWrite::Render,
    );
    // Claimed before the copy enters memory, so a memory hit
    // that learns a group meanwhile reaches the write.
    let disk = begin_disk_write(
        state,
        snapshot,
        cache_key,
        &entry,
        bucket_for(sent, &entry.groups),
    );
    let took = put_memory_tier(state, snapshot, cache_key, &entry);
    teach_slower_tiers(state, snapshot, cache_key, &entry, took);
    let Some(claim) = disk else {
        return;
    };
    if let Some(handoff) = handoff
        && state.cache.holds_copy(cache_key, &entry)
    {
        *handoff.lock().unwrap() = Some(PendingDiskWrite {
            snapshot,
            key: cache_key.to_string(),
            entry,
            claim,
        });
        return;
    }
    finish_disk_write(state, snapshot, cache_key, &entry, claim, false).await;
}
