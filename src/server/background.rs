//! The work handlers leave running after they answer
//! (issue #463), and the shutdown drain that waits for it.
//!
//! A shared-tier write-through, a purge whose caller has
//! gone, a disk hit's follow-up writes, and a purge's
//! background walk all outlive the request that started
//! them. Each one is spawned through [`Background`], so
//! none is a bare `tokio::spawn`: every one is tracked.
//! The write-throughs are bounded here too, since every
//! miss starts one and nothing else limits them; purges
//! are signed, and their walks wait for a turn (issue
//! #462); a disk hit's task is one disk read its request
//! waits for.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;

/// How many shared-tier write-throughs a process keeps in
/// flight (issue #463). Each one holds a whole rendered
/// entry until the bucket answers, so a slow or stalled
/// bucket would otherwise let them pile up with every
/// miss. A write that finds them all taken is skipped:
/// the local tiers already hold the entry, and the next
/// replica to miss renders it again and writes it then.
pub const SHARED_PUTS_AT_ONCE: usize = 64;

/// How long a stopping server drains, from the signal on,
/// when `CRABCROP_SHUTDOWN_GRACE_SECS` is unset (issue
/// #463): under the 10 seconds `docker stop` and Cloud Run
/// allow before they kill the process, so the drain ends
/// on its own deadline rather than on the platform's.
pub const DEFAULT_SHUTDOWN_GRACE_SECS: u64 = 8;

/// The detached work of one server: write-throughs, purges
/// and disk-hit follow-ups that a shutdown waits for, and
/// background purge walks that it does not. Clones share
/// everything.
#[derive(Clone, Debug)]
pub struct Background {
    /// Detached work a shutdown drain waits for, up to its
    /// deadline: shared-tier write-throughs, purge fan-outs
    /// whose request may have gone, and a disk hit's
    /// follow-up writes.
    writes: TaskTracker,
    /// Background purge walks (issue #462). A drain never
    /// waits for them: a walk cut short leaves its purge's
    /// bucket tombstone unended, which keeps the purge in
    /// force until a later purge of the same target
    /// finishes its walk (issue #461). They are tracked so
    /// that a shutdown can say how many it stopped.
    walks: TaskTracker,
    /// Write-through slots, [`SHARED_PUTS_AT_ONCE`] by
    /// default; a write holds one until the bucket has
    /// answered it and its reconcile is done.
    put_slots: Arc<Semaphore>,
    put_capacity: usize,
    /// Write-throughs skipped because every slot was taken.
    puts_skipped: Arc<AtomicU64>,
}

impl Default for Background {
    fn default() -> Self {
        Self::new(SHARED_PUTS_AT_ONCE)
    }
}

/// What a drain left running at its deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Drained {
    /// Tracked writes that had not finished: abandoned.
    pub writes: usize,
    /// Background purge walks, never waited for: stopped.
    pub walks: usize,
}

impl Background {
    /// Detached work with `put_capacity` write-through slots.
    pub fn new(put_capacity: usize) -> Self {
        Self {
            writes: TaskTracker::new(),
            walks: TaskTracker::new(),
            put_slots: Arc::new(Semaphore::new(put_capacity)),
            put_capacity,
            puts_skipped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run `task` detached, as work a shutdown drain waits
    /// for. Dropping the handle detaches the task; awaiting
    /// it is how a handler keeps its answer while the work
    /// finishes even if the caller goes (issue #347).
    pub(crate) fn spawn<F>(&self, task: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.writes.spawn(task)
    }

    /// Run a background purge walk: tracked, but never
    /// waited for at shutdown (see [`Background::drain`]).
    pub(crate) fn spawn_walk<F>(&self, task: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        drop(self.walks.spawn(task));
    }

    /// A write-through slot, or `None` with the skip
    /// counted when every slot is taken. Never waits: the
    /// write it guards is best-effort, and a request must
    /// not queue behind a stalled bucket.
    pub(crate) fn put_slot(&self) -> Option<OwnedSemaphorePermit> {
        let slot = self.put_slots.clone().try_acquire_owned().ok();
        if slot.is_none() {
            self.puts_skipped.fetch_add(1, Ordering::Relaxed);
        }
        slot
    }

    /// Write-throughs holding a slot right now.
    pub fn puts_in_flight(&self) -> usize {
        self.put_capacity - self.put_slots.available_permits()
    }

    /// The write-through slots there are.
    pub fn put_capacity(&self) -> usize {
        self.put_capacity
    }

    /// Write-throughs skipped so far for want of a slot.
    pub fn puts_skipped(&self) -> u64 {
        self.puts_skipped.load(Ordering::Relaxed)
    }

    /// Wait until the tracked writes have finished, or until
    /// `deadline`, whichever comes first, and report what is
    /// still running. Work spawned while the drain waits
    /// (by requests that are still finishing) is waited for
    /// too. Background purge walks are not: a walk cut
    /// short leaves its purge in force (issue #461), so
    /// they are only counted.
    pub async fn drain(&self, deadline: tokio::time::Instant) -> Drained {
        self.writes.close();
        let _ = tokio::time::timeout_at(deadline, self.writes.wait()).await;
        Drained {
            writes: self.writes.len(),
            walks: self.walks.len(),
        }
    }
}
