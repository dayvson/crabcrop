//! Disk-backed result storage: the persistent tier under the
//! in-memory result cache.
//!
//! Entries live one per file under a configured directory,
//! named by the SHA-256 of their cache key (keys contain
//! slashes and cannot be paths). A file is a small text header
//! (format, quality, and the original key, so purge-by-path
//! works after a restart) followed by the raw image bytes.
//! Writes go to a temporary file that is renamed into place,
//! so a process crash can leave an orphan temp file but never
//! a partial entry. Nothing is synced to disk first, so a
//! power loss or kernel crash can truncate an entry; the
//! header's length and MAC checks then fail it. A file that
//! fails to parse is a miss, is deleted, and is overwritten
//! by the next store. The byte budget evicts least recently
//! used entries, with recency tracked in memory and re-seeded
//! from file times at open. Each entry is charged its file
//! and the record the in-memory index keeps for it, so the
//! budget bounds the index's memory as well as the files.
//!
//! The index is the authority on an entry's cache groups
//! (issue #572). A group learned on a hit goes into the index
//! at once, with no file I/O, and a background drain writes it
//! into the entry file later. Purges remove index records at
//! once and the drain deletes their files. Only one writer or
//! deleter works on an entry file at a time: it holds the
//! key's claim.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::cache::{CachedImage, PurgeCutoff, PurgeTarget, Tombstone, WriteStamp};
use crate::url::ImageFormat;

type HmacSha256 = Hmac<Sha256>;

/// An index record (issue #572).
#[derive(Debug)]
struct Record {
    /// The entry's file stem.
    stem: String,
    /// The length of the file as last written.
    file_bytes: usize,
    /// What the entry is charged (see [`entry_charge`]).
    charge: usize,
    /// The entry's cache groups: every group the entry has
    /// been served under since it was indexed. Purges match
    /// on these. The file may lack some until the drain
    /// writes them.
    groups: Vec<String>,
    /// The copy's write stamp, for the purge checks a learn
    /// runs.
    written: Option<WriteStamp>,
    /// The copy's identity, fresh at scan and at every write:
    /// a read whose record changed meanwhile is a miss.
    id: u64,
    /// Groups were added while the bucket write that carries
    /// them was skipped (issue #463): the next disk or shared
    /// hit writes them back.
    owes_shared: bool,
}

/// Who works on a key's entry file right now (issue #572).
/// Only the holder renames into or unlinks the file.
#[derive(Debug)]
enum Claim {
    /// A fresh write in flight. The key is not indexed; the
    /// groups learned meanwhile land with it. A purge that
    /// covers it marks it doomed, and it then lands nothing.
    Write { groups: Vec<String>, doomed: bool },
    /// The drain or an inline delete owns the file.
    File,
}

/// One queued job for a key: a rewrite of its file's groups,
/// or the deletion of its file.
#[derive(Debug)]
struct Job {
    /// Order of queueing; a purge waits for its own range.
    seq: u64,
    /// The file to delete (deletions only).
    stem: String,
    /// Failed runs so far.
    attempts: u32,
    /// Not run again before this, after a failure: the job
    /// waits in [`KeyQueue::later`].
    not_before: Option<std::time::Instant>,
    /// Being run right now.
    running: bool,
}

/// Jobs by key, at most one per key. A job that can run waits
/// in `order`, in queueing order. One that failed waits out
/// its backoff in `later` instead, so a backlog that keeps
/// failing costs nothing until it is due. `order` can hold
/// stale entries: one whose job is gone, running, or has
/// another sequence number is dropped when it comes up, so
/// dropping a job costs no scan of the queue.
#[derive(Debug, Default)]
struct KeyQueue {
    order: VecDeque<(String, u64)>,
    later: BTreeMap<(std::time::Instant, u64), String>,
    jobs: HashMap<String, Job>,
    /// Every queued job's key by its sequence number, for the
    /// waits on one purge's deletions.
    by_seq: BTreeMap<u64, String>,
    /// Jobs running now.
    running: usize,
    /// Jobs that failed at least once.
    failing: usize,
    /// The sequence numbers of jobs that ended, failed or went
    /// since the waits on them last heard, with
    /// `track_ended` (deletions only).
    ended: Vec<u64>,
    track_ended: bool,
}

impl KeyQueue {
    /// Queue a job for `key` unless one is queued already.
    fn push(&mut self, key: &str, stem: &str, seq: &mut u64) {
        if self.jobs.contains_key(key) {
            return;
        }
        *seq += 1;
        self.jobs.insert(
            key.to_string(),
            Job {
                seq: *seq,
                stem: stem.to_string(),
                attempts: 0,
                not_before: None,
                running: false,
            },
        );
        self.by_seq.insert(*seq, key.to_string());
        self.order.push_back((key.to_string(), *seq));
    }

    /// Drop `key`'s job. Returns whether there was one.
    fn remove(&mut self, key: &str) -> bool {
        let Some(job) = self.jobs.remove(key) else {
            return false;
        };
        self.by_seq.remove(&job.seq);
        if self.track_ended {
            self.ended.push(job.seq);
        }
        if let Some(at) = job.not_before {
            self.later.remove(&(at, job.seq));
        }
        if job.running {
            self.running -= 1;
        }
        if job.attempts > 0 {
            self.failing -= 1;
        }
        true
    }

    /// Move the jobs whose backoff has passed, or all of them
    /// with `all`, back to the ready queue.
    fn wake_due(&mut self, now: std::time::Instant, all: bool) {
        while let Some(entry) = self.later.first_entry() {
            if !all && entry.key().0 > now {
                return;
            }
            let ((_, seq), key) = entry.remove_entry();
            if let Some(job) = self.jobs.get_mut(&key) {
                job.not_before = None;
            }
            self.order.push_back((key, seq));
        }
    }

    /// Take the next ready job off the queue: its key, its
    /// sequence number and its file stem. Stale entries are
    /// dropped on the way.
    fn pop_ready(&mut self) -> Option<(String, u64, String)> {
        while let Some((key, seq)) = self.order.pop_front() {
            if let Some(job) = self.jobs.get(&key)
                && job.seq == seq
                && !job.running
                && job.not_before.is_none()
            {
                let stem = job.stem.clone();
                return Some((key, seq, stem));
            }
        }
        None
    }

    /// Mark `key`'s job as running.
    fn start(&mut self, key: &str) {
        if let Some(job) = self.jobs.get_mut(key) {
            job.running = true;
            self.running += 1;
        }
    }

    /// End a running job: drop it, or keep it queued to run
    /// again at once.
    fn finish(&mut self, key: &str, again: bool) {
        match self.jobs.get_mut(key) {
            Some(job) if again && job.running => {
                job.running = false;
                self.running -= 1;
                if job.attempts > 0 {
                    job.attempts = 0;
                    self.failing -= 1;
                }
                self.order.push_back((key.to_string(), job.seq));
            }
            Some(_) => {
                self.remove(key);
            }
            None => {}
        }
    }

    /// Put a running job that failed back in the queue, to run
    /// again after a backoff.
    fn retry(&mut self, key: &str) {
        if let Some(job) = self.jobs.get_mut(key)
            && job.running
        {
            job.running = false;
            self.running -= 1;
            if job.attempts == 0 {
                self.failing += 1;
                if self.track_ended {
                    self.ended.push(job.seq);
                }
            }
            job.attempts = job.attempts.saturating_add(1);
            let backoff =
                std::time::Duration::from_secs(1 << job.attempts.min(5).saturating_sub(1))
                    .min(DRAIN_RETRY_CAP);
            let at = std::time::Instant::now() + backoff;
            job.not_before = Some(at);
            self.later.insert((at, job.seq), key.to_string());
        }
    }

    fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Jobs that can run now or once a claim is released: not
    /// running and not waiting out a backoff.
    fn ready(&self) -> usize {
        self.jobs
            .len()
            .saturating_sub(self.running + self.later.len())
    }

    /// Whether a job runs, or one that has not failed waits.
    fn busy(&self) -> bool {
        self.running > 0 || self.jobs.len() > self.failing
    }

    /// The earliest time a job waiting out a backoff is due.
    fn next_due(&self) -> Option<std::time::Instant> {
        self.later.first_key_value().map(|((at, _), _)| *at)
    }

    /// How many jobs in `seqs` are queued or running; failed
    /// ones count only with `failed`.
    fn count_in(&self, seqs: &Range<u64>, failed: bool) -> usize {
        self.by_seq
            .range(seqs.clone())
            .filter(|(_, key)| failed || self.jobs.get(*key).is_some_and(|job| job.attempts == 0))
            .count()
    }
}

/// What a background job does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobKind {
    Rewrite,
    Delete,
}

/// The allocator's bookkeeping charged for each separate heap
/// allocation an index record makes: its header and size
/// rounding.
const ALLOCATION_OVERHEAD: usize = 16;

/// The length of an entry's file stem: a hex SHA-256.
const STEM_LEN: usize = 64;

/// What an index record costs besides its strings: its slot in
/// each of the two maps, doubled for the spare slots a hash map
/// keeps as it grows, and the allocations behind its two key
/// copies and its file stem.
const RECORD_OVERHEAD: usize =
    2 * (size_of::<(String, Record)>() + size_of::<(String, u64)>()) + 3 * ALLOCATION_OVERHEAD;

/// What one entry is charged against the store's budget: its
/// file of `file_bytes`, plus the record the index keeps for
/// it in memory (the key twice, as it keys both maps, the file
/// stem, and each group name). Charging the file alone let
/// entries with tiny bodies under long keys hold about twice
/// the budget in index memory.
fn entry_charge(key: &str, file_bytes: usize, groups: &[String]) -> usize {
    let groups: usize = groups
        .iter()
        .map(|group| group.len() + size_of::<String>() + ALLOCATION_OVERHEAD)
        .sum();
    file_bytes + 2 * key.len() + STEM_LEN + groups + RECORD_OVERHEAD
}

#[derive(Debug, Default)]
struct Inner {
    /// Cache key -> its index record.
    entries: HashMap<String, Record>,
    /// Cache key -> who works on its file (issue #572).
    claims: HashMap<String, Claim>,
    /// Indexed keys whose file lacks groups the record has.
    rewrites: KeyQueue,
    /// Keys whose record is gone and whose file is still to
    /// be deleted.
    deletions: KeyQueue,
    /// The last record id handed out.
    next_id: u64,
    /// The last job sequence number handed out.
    next_seq: u64,
    /// Drain tasks running.
    workers: usize,
    /// Last-access stamp per key (monotonic clock). Eviction
    /// removes the smallest stamp, the least recently used
    /// entry: the same order the VecDeque kept, at O(1) per
    /// access instead of a linear scan per hit.
    stamps: HashMap<String, u64>,
    clock: u64,
    /// The sum of the indexed entries' charges.
    total_bytes: usize,
    /// Waits on one purge's deletions, by the first sequence
    /// number of its range: the range's end and the wait's
    /// wake. Purge ranges never overlap.
    deletion_waits: BTreeMap<u64, Vec<(u64, Arc<tokio::sync::Notify>)>>,
}

impl Inner {
    fn new() -> Self {
        Inner {
            deletions: KeyQueue {
                track_ended: true,
                ..KeyQueue::default()
            },
            ..Inner::default()
        }
    }

    /// Wake the waits whose purge had a deletion end, fail or
    /// go since they last heard (issue #572). Only those wake,
    /// so many waiting purges cost the drain nothing.
    fn wake_deletion_waits(&mut self) {
        for seq in self.deletions.ended.drain(..) {
            if let Some((_, waits)) = self.deletion_waits.range(..=seq).next_back() {
                for (end, wake) in waits {
                    if seq < *end {
                        wake.notify_one();
                    }
                }
            }
        }
    }
}

/// A byte-bounded, LRU-evicting file store. A `max_bytes` of 0
/// disables storage. All operations are best-effort: IO errors
/// degrade to misses and dropped writes, never failures,
/// because a cache must not take the pipeline down with it.
///
/// The mutex guards the in-memory index only (issue #137): file
/// reads, writes, and deletions all happen with the lock
/// dropped, so a slow disk never stalls unrelated lookups
/// behind it. The index mutations stay exactly as atomic as
/// before; a race can at worst evict a fresh entry, which a
/// cache heals on the next store.
pub struct DiskStore {
    dir: PathBuf,
    max_bytes: usize,
    /// The cache entry MAC key, derived from the server
    /// secret (issue #190): entries are written and
    /// verified under it, so an entry planted in the
    /// directory by anyone without the secret is a miss.
    entry_key: Vec<u8>,
    inner: Mutex<Inner>,
    /// Write failures, counted and logged as they happen
    /// (issue #367): a cache stays best-effort, but a tier
    /// that cannot write must be visible in the logs and
    /// on /metrics instead of failing silently forever.
    errors: std::sync::atomic::AtomicU64,
    /// Bounds how many store operations run on tokio's
    /// blocking pool at once (issue #347): the permit is
    /// taken on the async side, so operations waiting for
    /// one hold no thread, and a slow disk cannot crowd
    /// out the other work the pool carries.
    permits: Arc<tokio::sync::Semaphore>,
    /// Wakes a drain task that waits for a job (issue #572).
    drain_wake: tokio::sync::Notify,
    /// Bumped whenever a background job ends or a queued job
    /// goes, for the waits on them.
    progress: tokio::sync::watch::Sender<u64>,
    /// Rate-limits the log lines for disk waits that ran out.
    stall_log: StallLog,
    /// Rate-limits the log lines for background jobs that
    /// failed: a disk that refuses them fails each one again
    /// at every retry.
    job_log: StallLog,
    /// Test-only: run at every entry read (after the bytes are
    /// in, as a read stuck in the kernel has its file open),
    /// write and delete, on the blocking thread, with the
    /// file's path, so a test can hold or fail the
    /// operation the way a stuck or failing disk would. The
    /// store refuses FIFOs (issue #545), so a test cannot
    /// hold a read on one.
    #[cfg(test)]
    hold: Mutex<Option<TestHold>>,
}

/// A test hook for [`DiskStore`]'s file operations.
#[cfg(test)]
pub(crate) type TestHold = Arc<dyn Fn(OpKind, &Path) -> std::io::Result<()> + Send + Sync>;

/// The file operations a test can hold (issue #572).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpKind {
    /// An entry read: a lookup or a drain rewrite.
    Read,
    /// An entry write and its rename into place.
    Write,
    /// An entry file's deletion.
    Unlink,
}

impl std::fmt::Debug for DiskStore {
    /// The entry MAC key prints as `[redacted]` (issue #455).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskStore")
            .field("dir", &self.dir)
            .field("max_bytes", &self.max_bytes)
            .field("entry_key", &crate::redact::Redacted)
            .field("inner", &self.inner)
            .field("errors", &self.errors)
            .field("permits", &self.permits)
            .finish()
    }
}

/// How many disk operations may run on the blocking pool
/// at once (issue #347). Fixed, not configured: the disk
/// acceptance for this boundary is that waiting work
/// holds no thread, not a new tuning knob.
pub(crate) const DISK_OP_PERMITS: usize = 32;

/// How many background jobs (group rewrites and deletions)
/// run at once (issue #572). Claims keep them apart per key;
/// the rest of the slots stay for lookups and writes.
const DRAIN_JOBS: usize = DISK_OP_PERMITS / 4;

/// The longest wait before a failed background job runs
/// again.
const DRAIN_RETRY_CAP: std::time::Duration = std::time::Duration::from_secs(30);

/// How often a disk wait that ran out is logged at most
/// (issue #572). Each one still counts on /metrics.
const STALL_LOG_EVERY_MS: u64 = 10_000;

/// Entry format version. CRABDISK2 adds the entry's cache
/// groups to the header; CRABDISK3 marks the move to
/// canonical cache keys (Q1), a one-time invalidation of
/// entries keyed by spelling; CRABDISK4 adds an
/// HMAC-SHA256 over the entry under a key derived from
/// the server secret (issue #190, audit R2-13 part a),
/// so persistent entries are authenticated, not merely
/// self-consistent. CRABDISK5 adds the entry's ETag (a
/// digest of its payload bytes) to the header, so a hit
/// answers conditional requests without rehashing the
/// body. CRABDISK6 adds the entry's write stamp (issues
/// #446, #461: when the render behind it began, and on
/// which process), so a purge removes only the copies
/// written before it began. CRABDISK5 and CRABDISK4 files
/// still parse, as unstamped entries that count as older
/// than every purge, and a CRABDISK4 entry's ETag is
/// derived from its payload on read: entries written by
/// earlier builds keep serving instead of being dropped.
/// Anything older does not parse and is deleted on sight
/// at scan, the store's standing upgrade behavior for a
/// cache.
const MAGIC: &str = "CRABDISK6";

/// The entry format before write stamps: identical but
/// for the missing stamp line. Readable, never written.
const MAGIC_V5: &str = "CRABDISK5";

/// The entry format before stored ETags: CRABDISK5 without
/// the ETag line. Readable, never written.
const MAGIC_V4: &str = "CRABDISK4";

/// The largest serialized entry the shared (S3) tier will
/// read back, mirroring the disk tier's default total
/// budget. The disk tier caps reads at its own configured
/// budget instead: `put` there already refuses a single
/// entry larger than the budget, so a bigger file on disk
/// was never written by this build.
pub(crate) const MAX_ENTRY_BYTES: usize = 512 * 1024 * 1024;

/// Read a whole file, refusing anything past `cap` bytes.
/// Cache entries are operator-adjacent data, not fully
/// trusted input: a planted or corrupted oversized file
/// must not turn a lookup or the boot scan into an
/// unbounded allocation. `None` on IO error, an oversized
/// file, or a path that is not a regular file; all read as
/// a corrupt entry upstream.
fn read_capped(path: &Path, cap: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut file = open_regular(path)?;
    let mut data = Vec::new();
    file.by_ref()
        .take(cap.saturating_add(1) as u64)
        .read_to_end(&mut data)
        .ok()?;
    if data.len() > cap {
        return None;
    }
    Some(data)
}

/// Open `path` for reading if it is a regular file (issue
/// #545). Only `put` writes entries, and it writes regular
/// files, so anything else at an entry path was put there by
/// someone else. A FIFO is the dangerous case: a plain open
/// for reading waits in open(2) until a writer appears, which
/// may be never, and holds a blocking thread and a disk
/// permit all that time. So the open does not wait, and the
/// type is checked on the open file rather than on the path,
/// so a swap between the check and the open cannot slip one
/// through. The flag is cleared before returning, so reads
/// of a regular file behave as they always did. The open
/// does not follow a symlink either: the check would then
/// see the target, and the boot scan skips symlinks.
#[cfg(unix)]
fn open_regular(path: &Path) -> Option<std::fs::File> {
    use rustix::fs::{Mode, OFlags};
    let fd = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .ok()?;
    let file = std::fs::File::from(fd);
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let flags = rustix::fs::fcntl_getfl(&file).ok()?;
    rustix::fs::fcntl_setfl(&file, flags - OFlags::NONBLOCK).ok()?;
    Some(file)
}

/// Elsewhere there are no FIFOs whose open waits; the type
/// check still keeps reads to regular files.
#[cfg(not(unix))]
fn open_regular(path: &Path) -> Option<std::fs::File> {
    let file = std::fs::File::open(path).ok()?;
    file.metadata().ok()?.is_file().then_some(file)
}

/// The temp file a `put` writes before renaming into place.
/// The suffix is the process id plus a per-process counter:
/// the pid alone made every concurrent put of one key share
/// a single temp file, so two writers could interleave and
/// rename a torn write into place.
fn tmp_name(stem: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{stem}.tmp-{}-{n}", std::process::id())
}

/// One boot-time writability probe (issue #367): write a
/// scratch file into `dir` and remove it, so a directory
/// the process cannot write to (a read-only mount, an
/// unprivileged container user with no pre-created
/// directory) fails at store open with a clear message
/// instead of being discovered per request, silently, on
/// every put. The scratch name never matches the entry
/// scan's stem pattern and is removed on success anyway.
fn probe_writable(dir: &Path) -> Result<(), String> {
    let probe = dir.join(".crabcrop-write-probe");
    std::fs::write(&probe, b"").map_err(|e| {
        format!(
            "the result storage directory {} is not writable: {e}",
            dir.display()
        )
    })?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

impl DiskStore {
    /// Open (creating if needed) the store rooted at `dir`,
    /// indexing whatever entries a previous run left behind.
    /// `entry_key` is the cache entry MAC key (see
    /// [`crate::url::entry_key`]): files that fail
    /// verification under it are deleted on sight at scan.
    pub fn open(dir: &Path, max_bytes: usize, entry_key: Vec<u8>) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create the result storage directory: {e}"))?;
        probe_writable(dir)?;
        let store = Self {
            dir: dir.to_path_buf(),
            max_bytes,
            entry_key,
            inner: Mutex::new(Inner::new()),
            errors: std::sync::atomic::AtomicU64::new(0),
            permits: Arc::new(tokio::sync::Semaphore::new(DISK_OP_PERMITS)),
            drain_wake: tokio::sync::Notify::new(),
            progress: tokio::sync::watch::Sender::new(0),
            stall_log: StallLog::new("disk waits ran out"),
            job_log: StallLog::new("background disk jobs failed"),
            #[cfg(test)]
            hold: Mutex::new(None),
        };
        store.scan()?;
        Ok(store)
    }

    /// Write failures so far (issue #367), surfaced on
    /// /metrics as `crabcrop_cache_tier_errors_total`.
    pub fn errors(&self) -> u64 {
        self.errors.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Count and log one write failure. The store stays
    /// best-effort (the write is dropped, the pipeline
    /// continues), but never silently (issue #367).
    fn note_error(&self, what: &str, e: &std::io::Error) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!("result storage: {what}: {e}");
    }

    /// Count and log one failed run of a background job. It
    /// is retried; the log has the first failure at once, then
    /// at most one line every 10 seconds with a count.
    fn note_job_error(&self, what: &str, e: &std::io::Error) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(line) = self
            .job_log
            .note(crate::cache::now_ms(), &format!("{what}: {e}"))
        {
            eprintln!("{line}");
        }
    }

    /// Count and log a blocking task that died before it
    /// could report its operation's result (issue #367):
    /// the operation degrades the way a failed store call
    /// does, but the failure is on the record.
    pub(crate) fn note_task_failure(&self, what: &str) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!("result storage: disk task failed: {what}");
    }

    /// Whether the index holds `key`, without touching
    /// the filesystem: the cheap pre-check the request
    /// path runs before paying for the blocking boundary
    /// (issue #347). The index mutex is never held across
    /// file I/O, so this is safe on a runtime worker; a
    /// race with a concurrent write or eviction only
    /// turns a hit into a miss.
    pub fn contains(&self, key: &str) -> bool {
        self.max_bytes != 0 && self.inner.lock().unwrap().entries.contains_key(key)
    }

    /// Count and log a disk wait that ran out (issue #572):
    /// the request went on without the disk. Every one counts
    /// on /metrics; the log has the first at once, then at
    /// most one line every 10 seconds with a count.
    pub(crate) fn note_stall(&self, stall: Stall) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(line) = self.stall_log.note(crate::cache::now_ms(), stall.what()) {
            eprintln!("{line}");
        }
    }

    /// A slot for one store operation (issue #347), waited
    /// for with no limit: the caller bounds the wait. `None`
    /// cannot happen; the semaphore is never closed.
    pub async fn permit(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.permits.clone().acquire_owned().await.ok()
    }

    /// Store operations holding a slot right now.
    pub fn ops_in_flight(&self) -> usize {
        DISK_OP_PERMITS - self.permits.available_permits()
    }

    /// Run one store operation on tokio's blocking pool
    /// under `permit` and wait for its result (issue #347).
    /// The store is synchronous filesystem I/O; calling it
    /// from the request path directly would stall a runtime
    /// worker for the duration of the read or the write.
    /// `None` when the task died before reporting: the caller
    /// degrades to the operation's failure shape (a miss, a
    /// dropped write), the store's standing best-effort rule,
    /// with the failure counted through
    /// [`DiskStore::note_task_failure`]. The operation runs
    /// to its end even when the caller stops waiting: an
    /// operation stuck in the kernel cannot be cancelled. Work
    /// it queued is started on the blocking thread, so a
    /// caller that stopped waiting cannot lose it.
    pub async fn blocking_with<T, F>(
        self: &Arc<Self>,
        permit: tokio::sync::OwnedSemaphorePermit,
        op: F,
    ) -> Option<T>
    where
        F: FnOnce(&Arc<DiskStore>) -> T + Send + 'static,
        T: Send + 'static,
    {
        let store = self.clone();
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let out = op(&store);
            store.kick();
            out
        })
        .await
        {
            Ok(value) => Some(value),
            Err(e) => {
                self.note_task_failure(&e.to_string());
                None
            }
        }
    }

    /// [`DiskStore::blocking_with`] after waiting for a slot
    /// with no limit: for tests and work no request waits on.
    pub async fn blocking<T, F>(self: &Arc<Self>, op: F) -> Option<T>
    where
        F: FnOnce(&DiskStore) -> T + Send + 'static,
        T: Send + 'static,
    {
        let permit = self.permit().await?;
        self.blocking_with(permit, move |store| op(store)).await
    }

    /// Index existing entry files, oldest first, so the LRU
    /// order approximates the previous run's. Files that do
    /// not parse are deleted on sight. When the entries found
    /// are charged more than the budget, the oldest are
    /// deleted until the rest fit, so the budget holds from
    /// the first request.
    fn scan(&self) -> Result<(), String> {
        let mut found: Vec<Scanned> = Vec::new();
        let read = std::fs::read_dir(&self.dir)
            .map_err(|e| format!("cannot read the result storage directory: {e}"))?;
        for entry in read.flatten() {
            let path = entry.path();
            let Some(stem) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                .map(str::to_string)
            else {
                continue; // temp files and foreign files are ignored
            };
            if path.extension().and_then(|s| s.to_str()) != Some("bin") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            // Only `put` writes entries, and it writes regular
            // files (issue #545). Anything else here (a FIFO, a
            // socket, a device, a directory, a symlink) was put
            // there by someone else, and opening it can wait: a
            // FIFO holds open(2) until a writer appears, and its
            // length of 0 passes the size check below, so one
            // held the boot before the server listened. The
            // metadata comes from the directory entry and does
            // not follow a symlink, so nothing is opened. The
            // file is left in place, since the store did not
            // make it, and logged so the operator sees it.
            if !meta.is_file() {
                eprintln!(
                    "result storage: skipping {}: not a regular file",
                    path.display()
                );
                continue;
            }
            // `put` never writes an entry larger than the
            // budget, so a bigger file was not written by
            // this build: delete it on sight instead of
            // reading it into memory to find out.
            if self.max_bytes > 0 && meta.len() > self.max_bytes as u64 {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            // A zero budget disables the store (reads and
            // writes already refuse); the scan then imposes
            // no size cap of its own.
            let cap = if self.max_bytes == 0 {
                usize::MAX
            } else {
                self.max_bytes
            };
            match read_header(&path, cap, &self.entry_key) {
                Some((key, groups, written)) => {
                    let file_bytes = meta.len() as usize;
                    let charge = entry_charge(&key, file_bytes, &groups);
                    // `put` never stores an entry charged more
                    // than the whole budget. One here was
                    // stored under a larger budget or by a
                    // build that charged the file alone, and
                    // keeping it would leave no room for any
                    // other entry.
                    if self.max_bytes > 0 && charge > self.max_bytes {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    found.push(Scanned {
                        mtime: meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                        key,
                        stem,
                        file_bytes,
                        charge,
                        groups,
                        written,
                    });
                }
                None => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        found.sort_by_key(|scanned| scanned.mtime);
        // A directory filled under a larger budget, or by a
        // build that charged each entry its file alone, can
        // hold more than the budget. The oldest entries are
        // the ones `put` would evict first: delete them until
        // the rest fit, in one pass over the sorted list.
        let mut total: usize = found.iter().map(|scanned| scanned.charge).sum();
        let mut over = 0;
        if self.max_bytes > 0 {
            while total > self.max_bytes && over < found.len() {
                total -= found[over].charge;
                over += 1;
            }
        }
        for scanned in found.drain(..over) {
            let _ = std::fs::remove_file(self.file_for(&scanned.stem));
        }
        let mut inner = self.inner.lock().unwrap();
        for scanned in found {
            inner.total_bytes += scanned.charge;
            inner.clock += 1;
            let stamp = inner.clock;
            inner.stamps.insert(scanned.key.clone(), stamp);
            inner.next_id += 1;
            let id = inner.next_id;
            inner.entries.insert(
                scanned.key,
                Record {
                    stem: scanned.stem,
                    file_bytes: scanned.file_bytes,
                    charge: scanned.charge,
                    groups: scanned.groups,
                    written: scanned.written,
                    id,
                    owes_shared: false,
                },
            );
        }
        Ok(())
    }

    /// Run `hold` ahead of every entry read from now on, or
    /// stop with `None`.
    #[cfg(test)]
    pub(crate) fn set_read_hold(&self, hold: Option<Arc<dyn Fn() + Send + Sync>>) {
        self.set_hold(hold.map(|hold| -> TestHold {
            Arc::new(move |kind, _| {
                if kind == OpKind::Read {
                    hold();
                }
                Ok(())
            })
        }));
    }

    /// Run `hold` at every entry read, write and delete from
    /// now on, or stop with `None`. An `Err` fails that
    /// operation as the disk would.
    #[cfg(test)]
    pub(crate) fn set_hold(&self, hold: Option<TestHold>) {
        *self.hold.lock().unwrap() = hold;
    }

    /// The test hook for an operation of `kind`.
    #[cfg(test)]
    fn held(&self, kind: OpKind, path: &Path) -> std::io::Result<()> {
        // Cloned out first: the hook blocks, and the lock
        // must stay free for `set_hold`.
        let hold = self.hold.lock().unwrap().clone();
        match hold {
            Some(hold) => hold(kind, path),
            None => Ok(()),
        }
    }

    #[cfg(not(test))]
    fn held(&self, _kind: OpKind, _path: &Path) -> std::io::Result<()> {
        Ok(())
    }

    fn file_for(&self, stem: &str) -> PathBuf {
        self.dir.join(format!("{stem}.bin"))
    }

    /// Look up `key`, marking it most recently used. A corrupt
    /// or vanished file is a miss and is cleaned up, never
    /// served.
    pub fn get(&self, key: &str) -> Option<CachedImage> {
        let (image, id) = self.get_with_id(key)?;
        let mut inner = self.inner.lock().unwrap();
        if inner.entries.get(key).is_some_and(|record| record.id == id) {
            touch(&mut inner, key);
        }
        Some(image)
    }

    /// Read the copy indexed under `key`, with its record id.
    /// The copy carries every group its record holds, also
    /// ones its file has not caught up with. A read whose
    /// record was removed or replaced while it ran is a miss
    /// (issue #572): its bytes may be a purged copy's. A
    /// corrupt or vanished file is a miss and is cleaned up,
    /// never served. Recency is left alone: only a hit that
    /// serves counts as a use (see [`DiskStore::learn_hit`]).
    pub fn get_with_id(&self, key: &str) -> Option<(CachedImage, u64)> {
        if self.max_bytes == 0 {
            return None;
        }
        let (stem, id) = {
            let inner = self.inner.lock().unwrap();
            let record = inner.entries.get(key)?;
            (record.stem.clone(), record.id)
        };
        let path = self.file_for(&stem);
        let data = read_capped(&path, self.max_bytes);
        let parsed = match self.held(OpKind::Read, &path) {
            Ok(()) => data.and_then(|data| parse_entry(&data, &self.entry_key)),
            Err(_) => None,
        };
        let mut inner = self.inner.lock().unwrap();
        let record = inner.entries.get(key).filter(|record| record.id == id)?;
        match parsed {
            Some((stored_key, mut image)) if stored_key == key => {
                image.groups = with_groups(&image.groups, &record.groups);
                Some((image, id))
            }
            _ => {
                let stem = evict_entry(&mut inner, key).map(|record| record.stem);
                let claimed = stem.and_then(|stem| claim_or_queue(&mut inner, key, &stem));
                drop(inner);
                if let Some(stem) = claimed {
                    let _claim = FileClaim::inline(self, key);
                    self.unlink(&self.file_for(&stem));
                }
                None
            }
        }
    }

    /// The groups this process knows for `key`: its record's,
    /// or those of a write of it in flight that no purge
    /// doomed.
    pub fn known_groups(&self, key: &str) -> Option<Vec<String>> {
        if self.max_bytes == 0 {
            return None;
        }
        let inner = self.inner.lock().unwrap();
        match (inner.entries.get(key), inner.claims.get(key)) {
            (Some(record), _) => Some(record.groups.clone()),
            // A doomed write lands nothing: a purge covers its
            // groups.
            (
                None,
                Some(Claim::Write {
                    groups,
                    doomed: false,
                }),
            ) => Some(groups.clone()),
            _ => None,
        }
    }

    /// Record that a copy of `key` was served under `groups`
    /// (issue #572), with no file I/O: the index record learns
    /// them, or the write of the key in flight does. A record
    /// that grows is queued for the drain, which writes the
    /// groups into its file. `bucket` says what the bucket
    /// copy carries (see [`Bucket`]).
    pub fn learn_groups(&self, key: &str, groups: &[String], bucket: Bucket<'_>) -> Learned {
        if self.max_bytes == 0 {
            return Learned::Absent;
        }
        let mut inner = self.inner.lock().unwrap();
        learn_locked(&mut inner, key, groups, bucket)
    }

    /// [`DiskStore::learn_groups`] for the copy a disk hit
    /// read: `None` when the record is no longer the one read
    /// (`id`). A hit counts as a use. The record's debt to the
    /// bucket is cleared and returned in `owed`: the caller
    /// pays it with a write-back.
    pub fn learn_hit(&self, key: &str, id: u64, groups: &[String]) -> Option<HitLearn> {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        if !inner.entries.get(key).is_some_and(|record| record.id == id) {
            return None;
        }
        let learned = learn_locked(inner, key, groups, Bucket::Unknown);
        touch(inner, key);
        let record = inner.entries.get_mut(key)?;
        let owed = std::mem::take(&mut record.owes_shared);
        match learned {
            Learned::Indexed { grew, groups, .. } => Some(HitLearn { grew, owed, groups }),
            _ => None,
        }
    }

    /// Remove `key`'s record if it is still copy `id`, and
    /// queue its file's deletion. No file I/O; the caller
    /// kicks the drain. Returns whether it was removed.
    pub fn forget_if(&self, key: &str, id: u64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if !inner.entries.get(key).is_some_and(|record| record.id == id) {
            return false;
        }
        if let Some(record) = evict_entry(&mut inner, key) {
            let inner = &mut *inner;
            inner.deletions.push(key, &record.stem, &mut inner.next_seq);
        }
        true
    }

    /// Start a write of `key` (issue #572). A key the index
    /// holds is never written again: its record learns
    /// `groups` and the plan is [`WritePlan::Held`]. A key
    /// another write is in flight for joins it with `groups`.
    /// A key whose file the drain holds, or a write that may
    /// not claim (`may_claim` false: a purge covers it), is
    /// [`WritePlan::Busy`]. Otherwise the key is claimed: the
    /// write lands through [`DiskStore::land`], and dropping
    /// the claim unregisters it.
    pub fn begin_write(
        self: &Arc<Self>,
        key: &str,
        groups: &[String],
        bucket: Bucket<'_>,
        may_claim: bool,
    ) -> WritePlan {
        if self.max_bytes == 0 {
            return WritePlan::Off;
        }
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        if inner.entries.contains_key(key) {
            return WritePlan::Held(learn_locked(inner, key, groups, bucket));
        }
        match inner.claims.get_mut(key) {
            Some(Claim::Write { groups: held, .. }) => {
                *held = with_groups(held, groups);
                WritePlan::Joined
            }
            Some(Claim::File) => WritePlan::Busy,
            None if !may_claim => WritePlan::Busy,
            None => {
                inner.claims.insert(
                    key.to_string(),
                    Claim::Write {
                        groups: groups.to_vec(),
                        doomed: false,
                    },
                );
                WritePlan::Claimed(WriteClaim {
                    store: self.clone(),
                    key: key.to_string(),
                    landed: false,
                })
            }
        }
    }

    /// Land the write `claim` started: write `image` with
    /// every group learned meanwhile, index it, and evict
    /// least recently used entries until its charge fits. A
    /// write a purge doomed while it ran is deleted again and
    /// not indexed. `None` when nothing was indexed.
    pub fn land(&self, mut claim: WriteClaim, image: &CachedImage) -> Option<Landed> {
        claim.landed = true;
        self.land_claimed(&claim.key, image)
    }

    /// [`DiskStore::land`] for a key this store holds a write
    /// claim on. The claim goes on every path.
    fn land_claimed(&self, key: &str, image: &CachedImage) -> Option<Landed> {
        let release = ReleaseWrite { store: self, key };
        let entry = {
            let inner = self.inner.lock().unwrap();
            match inner.claims.get(key) {
                Some(Claim::Write {
                    groups,
                    doomed: false,
                }) if self.max_bytes != 0 => CachedImage {
                    groups: with_groups(&image.groups, groups),
                    ..image.clone()
                },
                _ => return None,
            }
        };
        let data = serialize_entry(key, &entry, &self.entry_key);
        if entry_charge(key, data.len(), &entry.groups) > self.max_bytes {
            return None;
        }
        let stem = key_stem(key);
        let tmp = self.dir.join(tmp_name(&stem));
        if let Err(e) = self
            .held(OpKind::Write, &tmp)
            .and_then(|()| std::fs::write(&tmp, &data))
        {
            self.note_error("cannot write a cache entry", &e);
            let _ = std::fs::remove_file(&tmp);
            return None;
        }
        if let Err(e) = std::fs::rename(&tmp, self.file_for(&stem)) {
            self.note_error("cannot move a cache entry into place", &e);
            let _ = std::fs::remove_file(&tmp);
            return None;
        }
        std::mem::forget(release);
        let mut evicted: Vec<(String, String)> = Vec::new();
        let landed = {
            let mut guard = self.inner.lock().unwrap();
            let inner = &mut *guard;
            // The rename replaced the file a queued deletion was
            // for.
            if inner.deletions.remove(key) {
                inner.wake_deletion_waits();
                self.progress.send_modify(|n| *n += 1);
            }
            match inner.claims.remove(key) {
                Some(Claim::Write {
                    groups,
                    doomed: false,
                }) => {
                    let groups = with_groups(&entry.groups, &groups);
                    if groups.len() > entry.groups.len() {
                        inner.rewrites.push(key, &stem, &mut inner.next_seq);
                    }
                    let charge = entry_charge(key, data.len(), &groups);
                    while inner.total_bytes + charge > self.max_bytes {
                        let oldest = inner
                            .stamps
                            .iter()
                            .min_by_key(|(_, stamp)| *stamp)
                            .map(|(k, _)| k.clone());
                        let Some(oldest) = oldest else { break };
                        if let Some(record) = evict_entry(inner, &oldest)
                            && let Some(stem) = claim_or_queue(inner, &oldest, &record.stem)
                        {
                            evicted.push((oldest, stem));
                        }
                    }
                    inner.next_id += 1;
                    let id = inner.next_id;
                    inner.total_bytes += charge;
                    inner.clock += 1;
                    let stamp = inner.clock;
                    inner.stamps.insert(key.to_string(), stamp);
                    inner.entries.insert(
                        key.to_string(),
                        Record {
                            stem: stem.clone(),
                            file_bytes: data.len(),
                            charge,
                            groups: groups.clone(),
                            written: entry.written,
                            id,
                            owes_shared: false,
                        },
                    );
                    Some(Landed { id, groups })
                }
                _ => {
                    // Doomed: the file just written goes again,
                    // under the key's claim.
                    inner.claims.insert(key.to_string(), Claim::File);
                    None
                }
            }
        };
        if landed.is_none() {
            let _claim = FileClaim::inline(self, key);
            self.unlink(&self.file_for(&stem));
        }
        for (old_key, old_stem) in evicted {
            let _claim = FileClaim::inline(self, &old_key);
            self.unlink(&self.file_for(&old_stem));
        }
        landed
    }

    /// Store `image` under `key` at once, replacing any copy
    /// held, and keeping every group the old record had.
    /// Evicts least recently used entries until the new
    /// entry's charge fits the budget. The server writes
    /// through [`DiskStore::begin_write`] instead.
    pub fn put(&self, key: &str, image: &CachedImage) {
        if self.max_bytes == 0 {
            return;
        }
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.claims.contains_key(key) {
                return;
            }
            let groups = evict_entry(&mut inner, key)
                .map(|record| record.groups)
                .unwrap_or_default();
            inner.claims.insert(
                key.to_string(),
                Claim::Write {
                    groups,
                    doomed: false,
                },
            );
        }
        self.land_claimed(key, image);
    }

    /// Remove `key`'s record if it is still copy `id`, and
    /// delete its file now, or queue the deletion if another
    /// job holds the file. Returns whether it was removed.
    pub fn remove_copy(&self, key: &str, id: u64) -> bool {
        let claimed = {
            let mut inner = self.inner.lock().unwrap();
            if !inner.entries.get(key).is_some_and(|record| record.id == id) {
                return false;
            }
            let record = evict_entry(&mut inner, key);
            record.and_then(|record| claim_or_queue(&mut inner, key, &record.stem))
        };
        if let Some(stem) = claimed {
            let _claim = FileClaim::inline(self, key);
            self.unlink(&self.file_for(&stem));
        }
        true
    }

    /// Remove the single entry stored under `key`, if
    /// any: its index record and its file. Returns whether
    /// an entry was removed.
    pub fn remove(&self, key: &str) -> bool {
        let id = {
            let inner = self.inner.lock().unwrap();
            match inner.entries.get(key) {
                Some(record) => record.id,
                None => return false,
            }
        };
        self.remove_copy(key, id)
    }

    /// Remove every record `target` covers from the index at
    /// once, with no file I/O (issue #572), and queue their
    /// files' deletion. A write in flight that the purge
    /// covers is doomed: it lands nothing. Returns how many
    /// records were removed, and the sequence numbers of the
    /// deletions this purge queued, for
    /// [`DiskStore::wait_deletions`]. The caller kicks the
    /// drain.
    pub fn purge(&self, target: &PurgeTarget) -> (usize, Range<u64>) {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        let first = inner.next_seq + 1;
        let removed: Vec<(String, Record)> = inner
            .entries
            .extract_if(|key, record| target.covers(key, &record.groups))
            .collect();
        for (key, record) in &removed {
            inner.total_bytes -= record.charge;
            inner.stamps.remove(key);
            if inner.rewrites.jobs.get(key).is_some_and(|job| !job.running) {
                inner.rewrites.remove(key);
            }
            inner.deletions.push(key, &record.stem, &mut inner.next_seq);
        }
        for (key, claim) in inner.claims.iter_mut() {
            if let Claim::Write { groups, doomed } = claim
                && target.covers(key, groups)
            {
                *doomed = true;
            }
        }
        (removed.len(), first..inner.next_seq + 1)
    }

    /// Remove every entry for `base`: the bare key plus its
    /// `#`-suffixed variants (Accept class, DPR). Returns how
    /// many entries were removed. Their files are deleted
    /// before it returns: for tests.
    #[cfg(test)]
    pub fn purge_matching(&self, base: &str) -> usize {
        let (removed, _) = self.purge(&PurgeTarget::Key(base.to_string()));
        self.drain_now();
        removed
    }

    /// Remove every entry belonging to `group` (RFC 9875
    /// grouped invalidation), as [`DiskStore::purge_matching`]
    /// does: for tests.
    #[cfg(test)]
    pub fn purge_group(&self, group: &str) -> usize {
        let (removed, _) = self.purge(&PurgeTarget::Group(group.to_string()));
        self.drain_now();
        removed
    }

    /// Remove every entry whose result derives from `source`,
    /// as [`DiskStore::purge_matching`] does: for tests.
    #[cfg(test)]
    pub fn purge_source(&self, source: &str) -> usize {
        let (removed, _) = self.purge(&PurgeTarget::Source(source.to_string()));
        self.drain_now();
        removed
    }

    /// Start background jobs for queued work (issue #572), up
    /// to [`DRAIN_JOBS`] at once, and wake one that waits. Jobs
    /// waiting out a backoff start no task: the last one left
    /// waits for them. Without a runtime the work stays queued.
    pub fn kick(self: &Arc<Self>) {
        let (spawn, ready) = {
            let mut inner = self.inner.lock().unwrap();
            let ready = inner.rewrites.ready() + inner.deletions.ready();
            let later = inner.rewrites.next_due().is_some() || inner.deletions.next_due().is_some();
            let spawn = inner.workers < DRAIN_JOBS
                && (ready > inner.workers || (inner.workers == 0 && later));
            if spawn {
                inner.workers += 1;
            }
            (spawn, ready > 0)
        };
        if ready {
            self.drain_wake.notify_one();
        }
        if spawn {
            match tokio::runtime::Handle::try_current() {
                Ok(runtime) => {
                    let store = self.clone();
                    drop(runtime.spawn(async move { store.drain().await }));
                }
                Err(_) => self.inner.lock().unwrap().workers -= 1,
            }
        }
    }

    /// One background drain task: take a disk slot, then a
    /// job, run it, and go on until no job is left. The slot
    /// comes first, so a job's claim is held only while its
    /// file operation can run. A job whose retry time has not
    /// come is waited for by the last task left.
    async fn drain(self: Arc<Self>) {
        let mut worker = Worker {
            store: &self,
            counted: true,
        };
        loop {
            let Some(permit) = self.permit().await else {
                return;
            };
            let (next, dropped) = {
                let mut inner = self.inner.lock().unwrap();
                let mut dropped = false;
                let next = next_job(&mut inner, false, &mut dropped);
                if matches!(next, Next::Empty) {
                    inner.workers -= 1;
                    worker.counted = false;
                }
                (next, dropped)
            };
            if dropped {
                self.progress.send_modify(|n| *n += 1);
            }
            match next {
                Next::Job(job) => {
                    let claim = FileClaim::job(self.clone(), &job.key, job.kind);
                    let _ = self
                        .blocking_with(permit, move |store| store.run_job(claim, &job))
                        .await;
                }
                Next::Wait(until) => {
                    drop(permit);
                    {
                        let mut inner = self.inner.lock().unwrap();
                        if inner.workers > 1 {
                            inner.workers -= 1;
                            worker.counted = false;
                            return;
                        }
                    }
                    let woken = self.drain_wake.notified();
                    match until {
                        Some(until) => {
                            let _ = tokio::time::timeout_at(until.into(), woken).await;
                        }
                        None => woken.await,
                    }
                }
                Next::Empty => return,
            }
        }
    }

    /// Run one job on a blocking thread, holding its claim.
    fn run_job<S: std::ops::Deref<Target = DiskStore>>(
        &self,
        mut claim: FileClaim<S>,
        job: &Taken,
    ) {
        let path = self.file_for(&job.stem);
        match job.kind {
            JobKind::Delete => {
                match self
                    .held(OpKind::Unlink, &path)
                    .and_then(|()| std::fs::remove_file(&path))
                {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        // The claim's drop queues it again.
                        self.note_job_error("cannot delete a purged cache entry", &e);
                        return;
                    }
                }
                self.inner.lock().unwrap().deletions.finish(&job.key, false);
                claim.done = true;
            }
            JobKind::Rewrite => {
                if self.rewrite(&job.key, &job.stem, job.id) {
                    claim.done = true;
                }
            }
        }
    }

    /// Write the record's groups into `key`'s file, keeping
    /// its bytes, ETag and stamp. Returns false when the job
    /// failed and runs again later.
    fn rewrite(&self, key: &str, stem: &str, id: u64) -> bool {
        let path = &self.file_for(stem);
        let data = read_capped(path, self.max_bytes);
        let parsed = match self.held(OpKind::Read, path) {
            Ok(()) => data.and_then(|data| parse_entry(&data, &self.entry_key)),
            Err(e) => {
                self.note_job_error("cannot read a cache entry to add its groups", &e);
                return false;
            }
        };
        let Some((_, mut image)) = parsed.filter(|(stored, _)| stored == key) else {
            // Corrupt or gone: a copy that cannot carry its
            // groups does not stay.
            self.drop_copy(key, id, path);
            return true;
        };
        let groups = {
            let mut inner = self.inner.lock().unwrap();
            match inner.entries.get(key).filter(|record| record.id == id) {
                Some(record) => record.groups.clone(),
                None => {
                    inner.rewrites.finish(key, false);
                    return true;
                }
            }
        };
        if groups.iter().all(|g| image.groups.contains(g)) {
            self.inner.lock().unwrap().rewrites.finish(key, false);
            return true;
        }
        image.groups = with_groups(&image.groups, &groups);
        let data = serialize_entry(key, &image, &self.entry_key);
        if entry_charge(key, data.len(), &image.groups) > self.max_bytes {
            self.drop_copy(key, id, path);
            return true;
        }
        let tmp = self.dir.join(tmp_name(stem));
        if let Err(e) = self
            .held(OpKind::Write, &tmp)
            .and_then(|()| std::fs::write(&tmp, &data))
            .and_then(|()| std::fs::rename(&tmp, path))
        {
            self.note_job_error("cannot add cache groups to a cache entry", &e);
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        match inner.entries.get_mut(key).filter(|record| record.id == id) {
            Some(record) => {
                record.file_bytes = data.len();
                let charge = entry_charge(key, record.file_bytes, &record.groups);
                inner.total_bytes = inner.total_bytes - record.charge + charge;
                record.charge = charge;
                let again = record.groups.iter().any(|g| !image.groups.contains(g));
                inner.rewrites.finish(key, again);
            }
            None => {
                // A purge or an eviction removed the record
                // while the file was rewritten: the file is
                // this job's, under its claim.
                inner.rewrites.finish(key, false);
                drop(guard);
                self.unlink(path);
            }
        }
        true
    }

    /// Evict copy `id` of `key` and delete its file, under the
    /// claim the caller holds.
    fn drop_copy(&self, key: &str, id: u64, path: &Path) {
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.entries.get(key).is_some_and(|record| record.id == id) {
                evict_entry(&mut inner, key);
            }
            inner.rewrites.finish(key, false);
        }
        self.unlink(path);
    }

    /// Delete an entry file under the claim the caller holds,
    /// best-effort: a file left behind by a failure is one the
    /// next boot indexes again or deletes as corrupt.
    fn unlink(&self, path: &Path) {
        let _ = self
            .held(OpKind::Unlink, path)
            .and_then(|()| std::fs::remove_file(path));
    }

    /// Run every queued job now, on this thread, retry times
    /// ignored: for tests, which may have no runtime.
    #[cfg(test)]
    pub(crate) fn drain_now(&self) {
        loop {
            let next = next_job(&mut self.inner.lock().unwrap(), true, &mut false);
            let Next::Job(job) = next else { return };
            let claim = FileClaim::job(self, &job.key, job.kind);
            self.run_job(claim, &job);
        }
    }

    /// Run the next queued job now, on this thread: for
    /// tests. Returns whether there was one.
    #[cfg(test)]
    pub(crate) fn run_one_job(&self) -> bool {
        let next = next_job(&mut self.inner.lock().unwrap(), true, &mut false);
        let Next::Job(job) = next else { return false };
        let claim = FileClaim::job(self, &job.key, job.kind);
        self.run_job(claim, &job);
        true
    }

    /// The keys the index holds: for tests.
    #[cfg(test)]
    pub(crate) fn keys(&self) -> Vec<String> {
        self.inner.lock().unwrap().entries.keys().cloned().collect()
    }

    /// Wait until the background jobs that have not failed
    /// are done, or until `deadline` (issue #572), and report
    /// what is left. A job that keeps failing is not waited
    /// for: it would hold every shutdown to its whole grace.
    pub async fn settle(self: &Arc<Self>, deadline: tokio::time::Instant) -> Backlog {
        self.kick();
        let mut progress = self.progress.subscribe();
        loop {
            let busy = {
                let inner = self.inner.lock().unwrap();
                inner.rewrites.busy() || inner.deletions.busy()
            };
            if !busy
                || tokio::time::timeout_at(deadline, progress.changed())
                    .await
                    .is_err()
            {
                return self.backlog();
            }
        }
    }

    /// How many deletions in `seqs` are queued or running;
    /// failed ones count only with `failed`. Looks at that
    /// range only.
    fn deletions_left(&self, seqs: &Range<u64>, failed: bool) -> usize {
        self.inner.lock().unwrap().deletions.count_in(seqs, failed)
    }

    /// Wait until the deletions a purge queued (`seqs`) are
    /// done (issue #572). Returns false when none of them
    /// finished for `idle`, or `cap` passed in all: a disk that
    /// stops answering, or a slow one. Deletions that failed
    /// are not waited for; they are retried and counted. Only
    /// these deletions wake the wait, so other jobs ending do
    /// not restart the idle time, and many waits cost the drain
    /// nothing.
    pub async fn wait_deletions(
        self: &Arc<Self>,
        seqs: Range<u64>,
        idle: std::time::Duration,
        cap: std::time::Duration,
    ) -> bool {
        let wake = Arc::new(tokio::sync::Notify::new());
        let mut left = {
            let mut inner = self.inner.lock().unwrap();
            let left = inner.deletions.count_in(&seqs, false);
            if left == 0 {
                return true;
            }
            inner
                .deletion_waits
                .entry(seqs.start)
                .or_default()
                .push((seqs.end, wake.clone()));
            left
        };
        let _registered = DeletionWaitEntry {
            store: self,
            start: seqs.start,
            wake: &wake,
        };
        self.kick();
        let started = tokio::time::Instant::now();
        let end = started + cap;
        let mut quiet_until = started + idle;
        loop {
            if tokio::time::timeout_at(quiet_until.min(end), wake.notified())
                .await
                .is_err()
            {
                return false;
            }
            let now_left = self.deletions_left(&seqs, false);
            if now_left == 0 {
                return true;
            }
            if now_left < left {
                left = now_left;
                quiet_until = tokio::time::Instant::now() + idle;
            }
        }
    }

    /// Whether a deletion in `seqs` is still queued, failed or
    /// not.
    pub fn deletions_pending(&self, seqs: &Range<u64>) -> bool {
        self.deletions_left(seqs, true) > 0
    }

    /// What background work is left (issue #572). Scans the
    /// queues and the index: for shutdown and tests, never
    /// for /metrics.
    pub fn backlog(&self) -> Backlog {
        let inner = self.inner.lock().unwrap();
        let failing = inner.rewrites.failing + inner.deletions.failing;
        Backlog {
            behind: inner.rewrites.len(),
            deletions: inner.deletions.len(),
            failing,
            owed: inner
                .entries
                .values()
                .filter(|record| record.owes_shared)
                .count(),
        }
    }

    /// Entries whose file lacks groups its record has, and
    /// deleted entries whose file is still there: queue
    /// lengths, cheap enough for every /metrics scrape.
    pub fn queued(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.rewrites.len(), inner.deletions.len())
    }

    /// (entries, total charge) currently indexed, for the boot
    /// log line.
    pub fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.entries.len(), inner.total_bytes)
    }
}

pub(crate) fn key_stem(key: &str) -> String {
    hex::encode(Sha256::digest(key.as_bytes()))
}

fn touch(inner: &mut Inner, key: &str) {
    inner.clock += 1;
    let stamp = inner.clock;
    if let Some(s) = inner.stamps.get_mut(key) {
        *s = stamp;
    }
}

/// Remove `key`'s record from the index, with its charge,
/// its recency and a rewrite queued for it.
fn evict_entry(inner: &mut Inner, key: &str) -> Option<Record> {
    let record = inner.entries.remove(key)?;
    inner.total_bytes -= record.charge;
    inner.stamps.remove(key);
    if inner.rewrites.jobs.get(key).is_some_and(|job| !job.running) {
        inner.rewrites.remove(key);
    }
    Some(record)
}

/// Claim `key`'s file for an inline delete and return its
/// stem, or, when another job holds it, queue the deletion.
fn claim_or_queue(inner: &mut Inner, key: &str, stem: &str) -> Option<String> {
    if inner.claims.contains_key(key) {
        inner.deletions.push(key, stem, &mut inner.next_seq);
        return None;
    }
    inner.claims.insert(key.to_string(), Claim::File);
    Some(stem.to_string())
}

/// `existing` with `extra` added. Unchanged, order and all,
/// when it holds them already; otherwise the sorted union
/// every tier stores.
fn with_groups(existing: &[String], extra: &[String]) -> Vec<String> {
    if extra.iter().all(|g| existing.contains(g)) {
        existing.to_vec()
    } else {
        crate::cache::union_groups(existing, extra)
    }
}

/// [`DiskStore::learn_groups`] under the index lock.
fn learn_locked(inner: &mut Inner, key: &str, groups: &[String], bucket: Bucket<'_>) -> Learned {
    if let Some(record) = inner.entries.get_mut(key) {
        let merged = with_groups(&record.groups, groups);
        let grew = merged.len() > record.groups.len();
        if grew {
            record.groups = merged;
            let charge = entry_charge(key, record.file_bytes, &record.groups);
            inner.total_bytes = inner.total_bytes - record.charge + charge;
            record.charge = charge;
            if matches!(bucket, Bucket::Skipped) {
                record.owes_shared = true;
            }
            inner.rewrites.push(key, &record.stem, &mut inner.next_seq);
            inner.clock += 1;
            let stamp = inner.clock;
            if let Some(s) = inner.stamps.get_mut(key) {
                *s = stamp;
            }
        }
        if let Bucket::Carries(carried) = bucket
            && record.groups.iter().all(|g| carried.contains(g))
        {
            record.owes_shared = false;
        }
        return Learned::Indexed {
            grew,
            groups: record.groups.clone(),
            written: record.written,
            id: record.id,
        };
    }
    if let Some(Claim::Write { groups: held, .. }) = inner.claims.get_mut(key) {
        *held = with_groups(held, groups);
        return Learned::InFlight;
    }
    Learned::Absent
}

/// What a drain task does next.
enum Next {
    /// Run this job; its key is claimed.
    Job(Taken),
    /// Nothing can run yet: wait until the earliest retry, or
    /// with `None` until a claim is released.
    Wait(Option<std::time::Instant>),
    /// Nothing is queued that is not running.
    Empty,
}

/// A job taken off a queue, its key claimed.
struct Taken {
    key: String,
    stem: String,
    /// The record a rewrite is for.
    id: u64,
    kind: JobKind,
}

/// Take the next job that can run, claiming its key:
/// deletions first, since a purge waits for them. A key
/// another job or write holds is passed over until released.
/// `now_for_all` runs jobs whose retry time has not come.
/// `dropped` is set when a queued deletion was dropped.
fn next_job(inner: &mut Inner, now_for_all: bool, dropped: &mut bool) -> Next {
    let next = take_next_job(inner, now_for_all, dropped);
    if *dropped {
        inner.wake_deletion_waits();
    }
    next
}

/// [`next_job`] before the waits on dropped deletions are
/// woken.
fn take_next_job(inner: &mut Inner, now_for_all: bool, dropped: &mut bool) -> Next {
    let now = std::time::Instant::now();
    let mut blocked = false;
    let Inner {
        entries,
        claims,
        rewrites,
        deletions,
        ..
    } = inner;
    for (queue, kind) in [
        (&mut *deletions, JobKind::Delete),
        (&mut *rewrites, JobKind::Rewrite),
    ] {
        queue.wake_due(now, now_for_all);
        for _ in 0..queue.order.len() {
            let Some((key, seq, stem)) = queue.pop_ready() else {
                break;
            };
            let (stem, id) = match (kind, entries.get(&key)) {
                (JobKind::Delete, None) => (stem, 0),
                (JobKind::Delete, Some(_)) => {
                    // Indexed again: a later write replaced the
                    // file.
                    queue.remove(&key);
                    *dropped = true;
                    continue;
                }
                (JobKind::Rewrite, Some(record)) => (record.stem.clone(), record.id),
                (JobKind::Rewrite, None) => {
                    queue.remove(&key);
                    continue;
                }
            };
            if claims.contains_key(&key) {
                blocked = true;
                queue.order.push_back((key, seq));
                continue;
            }
            queue.start(&key);
            claims.insert(key.clone(), Claim::File);
            return Next::Job(Taken {
                key,
                stem,
                id,
                kind,
            });
        }
    }
    let earliest = [deletions.next_due(), rewrites.next_due()]
        .into_iter()
        .flatten()
        .min();
    if earliest.is_some() || blocked {
        Next::Wait(earliest)
    } else {
        Next::Empty
    }
}

/// A key's file claim, released when dropped, on every path
/// a panic included. A job's claim that drops before the job
/// is done puts the job back in its queue to run again after
/// a backoff.
struct FileClaim<S: std::ops::Deref<Target = DiskStore>> {
    store: S,
    key: String,
    job: Option<JobKind>,
    done: bool,
}

impl<'s> FileClaim<&'s DiskStore> {
    /// The claim an inline delete took under the index lock.
    fn inline(store: &'s DiskStore, key: &str) -> Self {
        FileClaim {
            store,
            key: key.to_string(),
            job: None,
            done: true,
        }
    }
}

impl<S: std::ops::Deref<Target = DiskStore>> FileClaim<S> {
    /// The claim a background job took when it was taken.
    fn job(store: S, key: &str, kind: JobKind) -> Self {
        FileClaim {
            store,
            key: key.to_string(),
            job: Some(kind),
            done: false,
        }
    }
}

impl<S: std::ops::Deref<Target = DiskStore>> Drop for FileClaim<S> {
    fn drop(&mut self) {
        let store = &*self.store;
        {
            let mut inner = store
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if matches!(inner.claims.get(&self.key), Some(Claim::File)) {
                inner.claims.remove(&self.key);
            }
            match (self.job, self.done) {
                (Some(JobKind::Rewrite), false) => inner.rewrites.retry(&self.key),
                (Some(JobKind::Delete), false) => inner.deletions.retry(&self.key),
                _ => {}
            }
            inner.wake_deletion_waits();
        }
        if self.job.is_some() {
            store.progress.send_modify(|n| *n += 1);
        }
    }
}

/// Removes a wait from [`Inner::deletion_waits`] when it ends.
struct DeletionWaitEntry<'s> {
    store: &'s DiskStore,
    start: u64,
    wake: &'s Arc<tokio::sync::Notify>,
}

impl Drop for DeletionWaitEntry<'_> {
    fn drop(&mut self) {
        let mut inner = self
            .store
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(waits) = inner.deletion_waits.get_mut(&self.start) {
            waits.retain(|(_, wake)| !Arc::ptr_eq(wake, self.wake));
            if waits.is_empty() {
                inner.deletion_waits.remove(&self.start);
            }
        }
    }
}

/// Removes a write claim that did not land.
struct ReleaseWrite<'s> {
    store: &'s DiskStore,
    key: &'s str,
}

impl Drop for ReleaseWrite<'_> {
    fn drop(&mut self) {
        let mut inner = self
            .store
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if matches!(inner.claims.get(self.key), Some(Claim::Write { .. })) {
            inner.claims.remove(self.key);
        }
    }
}

/// Counts a drain task as running until it stops, on every
/// path.
struct Worker<'s> {
    store: &'s DiskStore,
    counted: bool,
}

impl Drop for Worker<'_> {
    fn drop(&mut self) {
        if self.counted {
            let mut inner = self
                .store
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.workers -= 1;
        }
    }
}

/// What the bucket copy carries, said by the caller of a
/// learn (issue #572).
#[derive(Debug, Clone, Copy)]
pub enum Bucket<'a> {
    /// A bucket write carrying these groups was spawned, or
    /// one is queued: a debt for them is paid.
    Carries(&'a [String]),
    /// The render's bucket write was skipped for want of a
    /// slot (issue #463): groups the record learns now are
    /// owed to the bucket.
    Skipped,
    /// Nothing is known: the debt stays as it is.
    Unknown,
}

/// What a learn did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Learned {
    /// The record learned: whether it grew, its groups now,
    /// the copy's stamp and its id.
    Indexed {
        grew: bool,
        groups: Vec<String>,
        written: Option<WriteStamp>,
        id: u64,
    },
    /// A write of the key in flight learned; it lands with
    /// the groups.
    InFlight,
    /// No copy of the key is indexed or being written.
    Absent,
}

/// What a disk hit's learn did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HitLearn {
    /// The record grew.
    pub grew: bool,
    /// The record owed the bucket groups (see
    /// [`Bucket::Skipped`]); the debt is now the caller's.
    pub owed: bool,
    /// The record's groups now.
    pub groups: Vec<String>,
}

/// How a write goes (see [`DiskStore::begin_write`]).
#[derive(Debug)]
pub enum WritePlan {
    /// The key is claimed for this write.
    Claimed(WriteClaim),
    /// The index holds the key; its record learned.
    Held(Learned),
    /// Another write of the key is in flight and learned.
    Joined,
    /// Another job holds the key's file, or the write may not
    /// claim it; nothing is written.
    Busy,
    /// The store keeps nothing (a budget of 0).
    Off,
}

/// A write's claim on a key the index does not hold (issue
/// #572). Groups learned for the key while it is held land
/// with the write. Dropped before it lands, it unregisters.
pub struct WriteClaim {
    store: Arc<DiskStore>,
    key: String,
    landed: bool,
}

impl std::fmt::Debug for WriteClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteClaim")
            .field("key", &self.key)
            .finish()
    }
}

impl Drop for WriteClaim {
    fn drop(&mut self) {
        if self.landed {
            return;
        }
        {
            let mut inner = self
                .store
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if matches!(inner.claims.get(&self.key), Some(Claim::Write { .. })) {
                inner.claims.remove(&self.key);
            }
        }
        // A deletion passed over while the key was claimed
        // can run now.
        self.store.kick();
    }
}

/// A landed write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landed {
    /// The new record's id.
    pub id: u64,
    /// The groups it landed with.
    pub groups: Vec<String>,
}

/// Background disk work left (see [`DiskStore::backlog`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Backlog {
    /// Entries whose file lacks groups its record has.
    pub behind: usize,
    /// Deleted entries whose file is still there.
    pub deletions: usize,
    /// Of those jobs, the ones that failed and wait to run
    /// again.
    pub failing: usize,
    /// Records whose groups the bucket copy may lack.
    pub owed: usize,
}

/// A disk wait that ran out (issue #572).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stall {
    /// A lookup found no free disk slot in time.
    LookupSlot,
    /// A lookup's read did not answer in time.
    LookupRead,
    /// A write found no free disk slot in time and was
    /// skipped.
    Write,
    /// A purge's deletions made no progress in time; they go
    /// on in the background.
    PurgeDelete,
}

impl Stall {
    fn what(self) -> &'static str {
        match self {
            Stall::LookupSlot => "a disk lookup found no free disk slot in time",
            Stall::LookupRead => "a disk lookup's read did not answer in time",
            Stall::Write => "a disk write found no free disk slot in time and was skipped",
            Stall::PurgeDelete => {
                "a purge's disk deletions stopped making progress; they go on in the background"
            }
        }
    }
}

/// A rate-limited log of one kind of disk trouble: disk waits
/// that ran out, or background jobs that failed. The first
/// line comes at once, then at most one every
/// [`STALL_LOG_EVERY_MS`], with how many were left out since
/// the last one.
#[derive(Debug)]
pub(crate) struct StallLog {
    next_line_ms: AtomicU64,
    since: AtomicU64,
    /// What the left-out ones are, after their count.
    rest: &'static str,
}

impl StallLog {
    pub(crate) fn new(rest: &'static str) -> Self {
        StallLog {
            next_line_ms: AtomicU64::new(0),
            since: AtomicU64::new(0),
            rest,
        }
    }

    /// Count one stall at `now_ms`, and return the line to
    /// log, if one is due.
    pub(crate) fn note(&self, now_ms: u64, what: &str) -> Option<String> {
        let next = self.next_line_ms.load(Ordering::Relaxed);
        if now_ms < next
            || self
                .next_line_ms
                .compare_exchange(
                    next,
                    now_ms + STALL_LOG_EVERY_MS,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_err()
        {
            self.since.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(match self.since.swap(0, Ordering::Relaxed) {
            0 => format!("result storage: {what}"),
            more => format!(
                "result storage: {what}; {more} more {} since the last line",
                self.rest
            ),
        })
    }
}

/// An entry the boot scan found.
struct Scanned {
    mtime: std::time::SystemTime,
    key: String,
    stem: String,
    file_bytes: usize,
    charge: usize,
    groups: Vec<String>,
    written: Option<WriteStamp>,
}

/// The disk deletions one purge queued (issue #572): what its
/// bucket tombstone waits for before it is marked ended, so a
/// restart that indexes those files again still finds the
/// purge in force.
#[derive(Debug, Clone)]
pub struct DeletionWait {
    pub store: Arc<DiskStore>,
    pub seqs: Range<u64>,
}

impl DeletionWait {
    /// Whether any of the deletions is still queued, failed or
    /// not.
    pub fn pending(&self) -> bool {
        self.store.deletions_pending(&self.seqs)
    }

    /// Wait while the deletions make progress (see
    /// [`DiskStore::wait_deletions`]), then say whether every
    /// one of them is done. One that failed is not.
    pub async fn finished(&self, idle: std::time::Duration, cap: std::time::Duration) -> bool {
        self.store
            .wait_deletions(self.seqs.clone(), idle, cap)
            .await;
        !self.pending()
    }
}

fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        ImageFormat::Gif => "gif",
        ImageFormat::Jxl => "jxl",
        ImageFormat::Avif => "avif",
        ImageFormat::Mp4 => "mp4",
        ImageFormat::WebM => "webm",
    }
}

fn parse_format(name: &str) -> Option<ImageFormat> {
    Some(match name {
        "jpeg" => ImageFormat::Jpeg,
        "png" => ImageFormat::Png,
        "webp" => ImageFormat::WebP,
        "gif" => ImageFormat::Gif,
        "jxl" => ImageFormat::Jxl,
        "avif" => ImageFormat::Avif,
        "mp4" => ImageFormat::Mp4,
        "webm" => ImageFormat::WebM,
        _ => return None,
    })
}

/// The entry MAC: HMAC-SHA256 under the entry key over
/// everything after the MAC line (the field lines, the
/// key, the groups block, and the payload). A stored
/// entry is therefore authenticated end to end: changing
/// any byte it carries, payload included, fails
/// verification.
fn entry_mac(entry_key: &[u8], covered: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(entry_key).expect("HMAC accepts any key length");
    mac.update(covered);
    mac.finalize().into_bytes().into()
}

fn verify_entry_mac(entry_key: &[u8], covered: &[u8], expected: &[u8; 32]) -> bool {
    let mut mac = HmacSha256::new_from_slice(entry_key).expect("HMAC accepts any key length");
    mac.update(covered);
    mac.verify_slice(expected).is_ok()
}

pub(crate) fn serialize_entry(key: &str, image: &CachedImage, entry_key: &[u8]) -> Vec<u8> {
    let quality = image
        .quality
        .map(|q| q.to_string())
        .unwrap_or_else(|| "-".to_string());
    // The groups block: per group, a decimal byte length, a
    // newline, the raw bytes, a newline. Length-prefixed like
    // the key, so any group name survives.
    let mut groups_block = Vec::new();
    for group in &image.groups {
        groups_block.extend_from_slice(group.len().to_string().as_bytes());
        groups_block.push(b'\n');
        groups_block.extend_from_slice(group.as_bytes());
        groups_block.push(b'\n');
    }
    // The MAC covers everything after its own line, so the
    // body is assembled first and the MAC computed over it.
    let head = format!(
        "{}\n{quality}\n{}\n{}\n{}\n{}\n{}\n",
        format_name(image.format),
        image.groups.len(),
        key.len(),
        image.bytes.len(),
        image.etag,
        stamp_line(image.written)
    );
    let mut body =
        Vec::with_capacity(head.len() + key.len() + 1 + groups_block.len() + image.bytes.len());
    body.extend_from_slice(head.as_bytes());
    body.extend_from_slice(key.as_bytes());
    body.push(b'\n');
    body.extend_from_slice(&groups_block);
    body.extend_from_slice(&image.bytes);
    let mac = entry_mac(entry_key, &body);
    let mac_hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
    let mut out = Vec::with_capacity(MAGIC.len() + 1 + mac_hex.len() + 1 + body.len());
    out.extend_from_slice(MAGIC.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(mac_hex.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(&body);
    out
}

/// The CRABDISK6 stamp line: `<unix-ms> <writer-hex>`, or
/// `-` for a copy promoted or merged from an unstamped
/// entry, which stays unstamped wherever it goes.
fn stamp_line(written: Option<WriteStamp>) -> String {
    match written {
        Some(stamp) => format!("{} {:016x}", stamp.at_ms, stamp.writer),
        None => "-".to_string(),
    }
}

/// Parse a stamp line. The field is authenticated by the
/// entry MAC like every other, so the checks here are
/// framing: an exact shape, nothing that could overflow.
fn parse_stamp_line(line: &[u8]) -> Option<Option<WriteStamp>> {
    if line == b"-" {
        return Some(None);
    }
    let text = std::str::from_utf8(line).ok()?;
    let (at, writer) = text.split_once(' ')?;
    if at.is_empty() || at.len() > 20 || !at.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if writer.len() != 16 || !writer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(Some(WriteStamp {
        at_ms: at.parse().ok()?,
        writer: u64::from_str_radix(writer, 16).ok()?,
    }))
}

/// Take the next `\n`-terminated line starting at `pos`,
/// advancing past it. Raw bytes throughout: the payload is
/// arbitrary binary and never goes near a UTF-8 check; only
/// the short header lines are decoded, one at a time.
fn take_line<'a>(data: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    let end = data[*pos..].iter().position(|&b| b == b'\n')? + *pos;
    let line = &data[*pos..end];
    *pos = end + 1;
    Some(line)
}

/// A stored file's parsed header.
struct EntryHeader {
    key: String,
    format: ImageFormat,
    quality: Option<u8>,
    groups: Vec<String>,
    payload_offset: usize,
    /// The stored MAC, and where the region it covers
    /// (everything after the MAC line) begins.
    mac: [u8; 32],
    covered_start: usize,
    /// The entry's stored ETag (CRABDISK5 on). CRABDISK4
    /// entries carry none: `None`, and the reader derives
    /// the value from the payload instead.
    etag: Option<String>,
    /// The entry's write stamp (CRABDISK6); `None` for an
    /// unstamped entry, of any version.
    written: Option<WriteStamp>,
}

/// What the shared tier's purge walk reads from an entry
/// header: its claimed key, groups, and write stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HeaderProbe {
    pub key: String,
    pub groups: Vec<String>,
    pub written: Option<WriteStamp>,
}

/// Split a stored file into (key, format, quality, groups,
/// payload offset). `None` on any inconsistency, including a
/// payload whose length disagrees with the header.
/// `total_len` is the entry's full length: for a whole
/// entry in memory it is `data.len()`; for a header prefix
/// read on its own (the shared tier's purge walk) it is the
/// object's real length, and `data` then holds only the
/// header bytes, which is all this function touches. The
/// MAC is parsed but not verified here: verification needs
/// the whole entry, payload included, and happens in
/// [`parse_entry`] and [`read_header`], the serving paths.
fn parse_header_bytes(data: &[u8], total_len: usize) -> Option<EntryHeader> {
    let mut pos = 0;
    // (carries an ETag line, carries a stamp line)
    let (has_etag, has_stamp) = match take_line(data, &mut pos)? {
        line if line == MAGIC.as_bytes() => (true, true),
        line if line == MAGIC_V5.as_bytes() => (true, false),
        line if line == MAGIC_V4.as_bytes() => (false, false),
        _ => return None,
    };
    let mac_line = take_line(data, &mut pos)?;
    if mac_line.len() != 64 || !mac_line.iter().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mac_vec = hex_decode(mac_line)?;
    let mut mac = [0u8; 32];
    mac.copy_from_slice(&mac_vec);
    let covered_start = pos;
    let format_line = std::str::from_utf8(take_line(data, &mut pos)?).ok()?;
    let format = parse_format(format_line)?;
    let quality_line = std::str::from_utf8(take_line(data, &mut pos)?).ok()?;
    let quality = if quality_line == "-" {
        None
    } else {
        Some(quality_line.parse().ok()?)
    };
    let group_count: usize = std::str::from_utf8(take_line(data, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;
    let key_len: usize = std::str::from_utf8(take_line(data, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;
    let payload_len: usize = std::str::from_utf8(take_line(data, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;
    // The ETag line: a quoted 32-digit hex digest. The MAC
    // authenticates it like every other field, so the shape
    // check here is framing, not trust.
    let etag = if has_etag {
        let line = take_line(data, &mut pos)?;
        if line.len() != 34
            || line.first() != Some(&b'"')
            || line.last() != Some(&b'"')
            || !line[1..33].iter().all(|b| b.is_ascii_hexdigit())
        {
            return None;
        }
        Some(String::from_utf8(line.to_vec()).ok()?)
    } else {
        None
    };
    let written = if has_stamp {
        parse_stamp_line(take_line(data, &mut pos)?)?
    } else {
        None
    };
    // Every offset below is checked arithmetic: the header
    // fields are untrusted counts, and a length near
    // usize::MAX must fail the parse, not wrap or panic.
    let key_end = pos.checked_add(key_len)?;
    let key_bytes = data.get(pos..key_end)?;
    if data.get(key_end) != Some(&b'\n') {
        return None;
    }
    let key = String::from_utf8(key_bytes.to_vec()).ok()?;
    let mut cursor = key_end + 1;
    // Each group occupies at least three bytes on disk (a
    // one-digit length line, its newline, and the newline
    // after a zero-length name), so a count that cannot fit
    // in the bytes remaining is corrupt. The check runs
    // before the vector grows, and the vector then grows
    // only as groups actually parse.
    if group_count > data.len().saturating_sub(cursor) / 3 {
        return None;
    }
    let mut groups = Vec::new();
    for _ in 0..group_count {
        let len_line = std::str::from_utf8(take_line(data, &mut cursor)?).ok()?;
        let glen: usize = len_line.parse().ok()?;
        let gend = cursor.checked_add(glen)?;
        let gbytes = data.get(cursor..gend)?;
        if data.get(gend) != Some(&b'\n') {
            return None;
        }
        groups.push(String::from_utf8(gbytes.to_vec()).ok()?);
        cursor = gend + 1;
    }
    let payload_offset = cursor;
    if payload_offset.checked_add(payload_len) != Some(total_len) {
        return None;
    }
    Some(EntryHeader {
        key,
        format,
        quality,
        groups,
        payload_offset,
        mac,
        covered_start,
        etag,
        written,
    })
}

/// Decode lowercase hex (the MAC line's spelling, written
/// by [`serialize_entry`]; the parser's caller has already
/// checked length and charset).
fn hex_decode(line: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(line.len() / 2);
    for pair in line.as_chunks::<2>().0 {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

pub(crate) fn parse_entry(data: &[u8], entry_key: &[u8]) -> Option<(String, CachedImage)> {
    let header = parse_header_bytes(data, data.len())?;
    // The entry is authenticated before anything in it is
    // trusted (audit R2-13 part a): a forged or tampered
    // entry is a miss, exactly like a corrupt one.
    if !verify_entry_mac(entry_key, &data[header.covered_start..], &header.mac) {
        return None;
    }
    let payload = data.get(header.payload_offset..)?;
    if payload.is_empty() {
        return None;
    }
    // A CRABDISK4 entry carries no stored ETag: derive it
    // from the payload in hand, which is exactly the value
    // the stored line would have carried.
    let etag = match header.etag {
        Some(etag) => etag,
        None => crate::cache::etag_for_bytes(payload),
    };
    Some((
        header.key,
        CachedImage {
            bytes: bytes::Bytes::copy_from_slice(payload),
            format: header.format,
            quality: header.quality,
            groups: header.groups,
            etag,
            written: header.written,
        },
    ))
}

/// Just the key, groups, and stamp of a stored entry,
/// WITHOUT MAC verification, for the shared tier's purge
/// walk: matching a purge target needs the entry's claimed
/// key, groups, and write stamp, and verification needs
/// the whole object, which is exactly the full read the
/// walk exists to avoid. A forged header can at worst steer
/// a purge (delete a copy, or leave one for the tombstone
/// to keep missing); it can never be served, because every
/// serving read goes through [`parse_entry`] and verifies.
/// `total_len` is the object's real length; `data` holds at
/// least its header.
pub(crate) fn parse_entry_header(data: &[u8], total_len: usize) -> Option<HeaderProbe> {
    let header = parse_header_bytes(data, total_len)?;
    Some(HeaderProbe {
        key: header.key,
        groups: header.groups,
        written: header.written,
    })
}

/// Just the header of a stored file, for the boot scan.
/// The read is capped like any other entry read: the
/// header parse validates against the whole file, so the
/// file has to be in memory, but never more than `cap`.
/// The MAC is verified: a file planted in the directory
/// is dropped at scan, not indexed.
fn read_header(
    path: &Path,
    cap: usize,
    entry_key: &[u8],
) -> Option<(String, Vec<String>, Option<WriteStamp>)> {
    let data = read_capped(path, cap)?;
    let header = parse_header_bytes(&data, data.len())?;
    if !verify_entry_mac(entry_key, &data[header.covered_start..], &header.mac) {
        return None;
    }
    Some((header.key, header.groups, header.written))
}

/// The purge tombstone format (issue #461): what a purge
/// writes to the shared bucket so every replica, and the
/// same one after a restart, keeps treating the copies it
/// covers as purged until its walk has removed them. A
/// small text record, authenticated like an entry:
///
/// ```text
/// CRABPURGE1
/// <HMAC-SHA256 hex over everything below>
/// <key|group|source>
/// <purge start, unix ms> <writer, 16 hex digits>
/// <walk end, unix ms, or ->
/// <target length>
/// <target bytes>
/// ```
///
/// The MAC key is derived from the entry key under its own
/// label, so neither record can stand in for the other,
/// and a tombstone planted by anyone without the secret is
/// ignored.
const TOMBSTONE_MAGIC: &str = "CRABPURGE1";

/// The largest tombstone object a refresh reads. Targets
/// are cache keys, group names, and source locators, far
/// smaller than this.
pub(crate) const MAX_TOMBSTONE_BYTES: usize = 64 * 1024;

/// The tombstone MAC key: the entry key under a label of
/// its own.
fn tombstone_mac_key(entry_key: &[u8]) -> [u8; 32] {
    entry_mac(entry_key, b"crabcrop purge tombstone v1")
}

fn target_parts(target: &PurgeTarget) -> (&'static str, &str) {
    match target {
        PurgeTarget::Key(base) => ("key", base),
        PurgeTarget::Group(group) => ("group", group),
        PurgeTarget::Source(source) => ("source", source),
    }
}

pub(crate) fn serialize_tombstone(tombstone: &Tombstone, entry_key: &[u8]) -> Vec<u8> {
    let (kind, name) = target_parts(&tombstone.target);
    let ended = tombstone
        .ended_ms
        .map(|ms| ms.to_string())
        .unwrap_or_else(|| "-".to_string());
    let start = stamp_line(Some(WriteStamp {
        at_ms: tombstone.cutoff.start_ms,
        writer: tombstone.cutoff.writer,
    }));
    let mut body = format!("{kind}\n{start}\n{ended}\n{}\n", name.len()).into_bytes();
    body.extend_from_slice(name.as_bytes());
    let mac = entry_mac(&tombstone_mac_key(entry_key), &body);
    let mac_hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
    let mut out = format!("{TOMBSTONE_MAGIC}\n{mac_hex}\n").into_bytes();
    out.extend_from_slice(&body);
    out
}

/// Parse and authenticate a tombstone object. `None` for
/// anything that is not a whole, verified record: the
/// object's bytes come from the bucket, which is
/// operator-adjacent data, not trusted input.
pub(crate) fn parse_tombstone(data: &[u8], entry_key: &[u8]) -> Option<Tombstone> {
    if data.len() > MAX_TOMBSTONE_BYTES {
        return None;
    }
    let mut pos = 0;
    if take_line(data, &mut pos)? != TOMBSTONE_MAGIC.as_bytes() {
        return None;
    }
    let mac_line = take_line(data, &mut pos)?;
    if mac_line.len() != 64 || !mac_line.iter().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mac_vec = hex_decode(mac_line)?;
    let mut mac = [0u8; 32];
    mac.copy_from_slice(&mac_vec);
    if !verify_entry_mac(&tombstone_mac_key(entry_key), &data[pos..], &mac) {
        return None;
    }
    let kind = take_line(data, &mut pos)?;
    let start = parse_stamp_line(take_line(data, &mut pos)?)??;
    let ended_line = take_line(data, &mut pos)?;
    let ended_ms = if ended_line == b"-" {
        None
    } else {
        if ended_line.is_empty()
            || ended_line.len() > 20
            || !ended_line.iter().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        Some(std::str::from_utf8(ended_line).ok()?.parse().ok()?)
    };
    let len_line = take_line(data, &mut pos)?;
    if len_line.is_empty() || len_line.len() > 20 || !len_line.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let len: usize = std::str::from_utf8(len_line).ok()?.parse().ok()?;
    if pos.checked_add(len)? != data.len() {
        return None;
    }
    let name = String::from_utf8(data[pos..].to_vec()).ok()?;
    let target = match kind {
        b"key" => PurgeTarget::Key(name),
        b"group" => PurgeTarget::Group(name),
        b"source" => PurgeTarget::Source(name),
        _ => return None,
    };
    Some(Tombstone {
        target,
        cutoff: PurgeCutoff {
            start_ms: start.at_ms,
            writer: start.writer,
        },
        ended_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_write_probe_accepts_writable_and_rejects_unwritable_dirs() {
        // Issue #367: the probe an open() runs so an
        // unwritable directory fails at boot. /proc is not
        // writable even for a privileged process, which
        // makes it the portable unwritable fixture.
        let dir = temp_dir("probe");
        std::fs::create_dir_all(&dir).unwrap();
        probe_writable(&dir).unwrap();
        assert!(
            !dir.join(".crabcrop-write-probe").exists(),
            "the probe cleans up after itself"
        );
        let err = probe_writable(Path::new("/proc/self")).unwrap_err();
        assert!(err.contains("not writable"), "probe error: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn temp_names_are_unique_per_put() {
        // Concurrent puts of one key must never share a
        // temp file (review P-6).
        assert_ne!(tmp_name("abc"), tmp_name("abc"));
        assert!(tmp_name("abc").starts_with("abc.tmp-"));
    }

    fn temp_dir(tag: &str) -> crate::test_support::ScratchDir {
        crate::test_support::scratch_dir(&format!("store-{tag}"))
    }

    /// The entry MAC key the test stores run under.
    fn test_entry_key() -> Vec<u8> {
        crate::url::entry_key(b"storage-test-secret")
    }

    fn image(bytes: &[u8]) -> CachedImage {
        CachedImage {
            bytes: bytes::Bytes::copy_from_slice(bytes),
            format: ImageFormat::Png,
            quality: Some(80),
            groups: Vec::new(),
            etag: crate::cache::etag_for_bytes(bytes),
            written: None,
        }
    }

    fn grouped_image(bytes: &[u8], groups: &[&str]) -> CachedImage {
        CachedImage {
            groups: groups.iter().map(|g| g.to_string()).collect(),
            ..image(bytes)
        }
    }

    #[test]
    fn roundtrip_and_persistence_across_instances() {
        let dir = temp_dir("persist");
        // A payload with bytes that are not valid UTF-8, like
        // every real image format: the parser must treat the
        // payload as opaque binary.
        let img = image(&[0x89, 0x50, 0x4E, 0x47, 0xFF, 0xFE, 0x00, 0x80, 0xC3, 0x28]);
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("300x200/http://x/a.png", &img);
            assert_eq!(store.get("300x200/http://x/a.png"), Some(img.clone()));
        }
        // A fresh instance over the same directory still has it,
        // format and quality intact.
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(store.get("300x200/http://x/a.png"), Some(img));
        assert_eq!(store.stats().0, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_drops_exactly_one_entry() {
        let dir = temp_dir("remove");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("a", &image(b"one"));
        store.put("a#avif", &image(b"two"));
        assert!(store.remove("a"));
        assert_eq!(store.get("a"), None);
        // The variant is a different key: untouched.
        assert!(store.get("a#avif").is_some());
        assert!(!store.remove("a"));
        assert!(!store.remove("missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Serialize the way the previous build did: the
    /// CRABDISK4 layout, the current body minus the ETag
    /// line, MAC'd the same way. Only group-free entries:
    /// that is all the compatibility test stores.
    fn serialize_entry_v4(key: &str, image: &CachedImage, entry_key: &[u8]) -> Vec<u8> {
        serialize_legacy_entry(MAGIC_V4, key, image, entry_key)
    }

    /// An entry in a pre-stamp layout: CRABDISK4 (no ETag
    /// line) or CRABDISK5 (an ETag line, no stamp line).
    fn serialize_legacy_entry(
        magic: &str,
        key: &str,
        image: &CachedImage,
        entry_key: &[u8],
    ) -> Vec<u8> {
        assert!(image.groups.is_empty());
        let quality = image
            .quality
            .map(|q| q.to_string())
            .unwrap_or_else(|| "-".to_string());
        let etag_line = if magic == MAGIC_V5 {
            format!("{}\n", image.etag)
        } else {
            String::new()
        };
        let head = format!(
            "{}\n{quality}\n{}\n{}\n{}\n{etag_line}",
            format_name(image.format),
            image.groups.len(),
            key.len(),
            image.bytes.len()
        );
        let mut body = Vec::new();
        body.extend_from_slice(head.as_bytes());
        body.extend_from_slice(key.as_bytes());
        body.push(b'\n');
        body.extend_from_slice(&image.bytes);
        let mac = entry_mac(entry_key, &body);
        let mac_hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        let mut out = Vec::new();
        out.extend_from_slice(magic.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(mac_hex.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn previous_format_entries_parse_with_a_derived_etag() {
        let key = test_entry_key();
        let img = image(b"payload");
        let data = serialize_entry_v4("k", &img, &key);
        // Direct parse: the entry serves, with the ETag the
        // payload derives (the helper etags by the same
        // rule, so the whole entry compares equal).
        let (stored_key, parsed) = parse_entry(&data, &key).unwrap();
        assert_eq!(stored_key, "k");
        assert_eq!(parsed, img);
        // Through the store: the boot scan indexes the file
        // instead of deleting it, and the entry serves.
        let dir = temp_dir("v4");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{}.bin", key_stem("k"))), &data).unwrap();
        let store = DiskStore::open(&dir, 1_000_000, key).unwrap();
        assert_eq!(store.stats().0, 1);
        assert_eq!(store.get("k"), Some(img));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stamped_entries_round_trip_and_older_formats_read_unstamped() {
        // Issues #446, #461: the write stamp is persisted
        // and survives a reopen; an entry written before
        // stamps existed still serves, unstamped, which a
        // purge treats as older than itself.
        let key = test_entry_key();
        let stamp = WriteStamp {
            at_ms: 1_762_000_000_123,
            writer: 0x0123_4567_89ab_cdef,
        };
        let stamped = CachedImage {
            written: Some(stamp),
            ..image(b"stamped")
        };
        let data = serialize_entry("s", &stamped, &key);
        assert!(data.starts_with(b"CRABDISK6\n"));
        assert_eq!(parse_entry(&data, &key).unwrap().1, stamped);
        let probe = parse_entry_header(&data, data.len()).unwrap();
        assert_eq!(probe.key, "s");
        assert_eq!(probe.written, Some(stamp));
        // An unstamped copy stays unstamped when rewritten.
        let unstamped = serialize_entry("u", &image(b"plain"), &key);
        assert_eq!(parse_entry(&unstamped, &key).unwrap().1.written, None);
        let dir = temp_dir("stamps");
        {
            let store = DiskStore::open(&dir, 1_000_000, key.clone()).unwrap();
            store.put("s", &stamped);
        }
        for (name, magic) in [("v5", MAGIC_V5), ("v4", MAGIC_V4)] {
            let legacy = serialize_legacy_entry(magic, name, &image(b"legacy"), &key);
            let (_, parsed) = parse_entry(&legacy, &key).unwrap();
            assert_eq!(parsed, image(b"legacy"), "{name}");
            assert_eq!(
                parse_entry_header(&legacy, legacy.len()).unwrap().written,
                None
            );
            std::fs::write(dir.join(format!("{}.bin", key_stem(name))), &legacy).unwrap();
        }
        let store = DiskStore::open(&dir, 1_000_000, key).unwrap();
        assert_eq!(store.stats().0, 3);
        assert_eq!(store.get("s").unwrap().written, Some(stamp));
        assert_eq!(store.get("v5").unwrap().written, None);
        assert_eq!(store.get("v4").unwrap().written, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stamp_lines_it_cannot_trust_fail_the_parse() {
        // The stamp line is MACed, but framing is checked
        // before anything is trusted: every malformed shape
        // is a miss, never a panic or a wrapped number.
        let key = test_entry_key();
        let with_stamp = |line: &str| {
            let mut body =
                format!("png\n-\n0\n1\n1\n\"00000000000000000000000000000000\"\n{line}\nk\nx")
                    .into_bytes();
            let mac = entry_mac(&key, &body);
            let mac_hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
            let mut v = format!("{MAGIC}\n{mac_hex}\n").into_bytes();
            v.append(&mut body);
            v
        };
        let good = parse_entry(&with_stamp("5 00000000000000ff"), &key).unwrap();
        assert_eq!(
            good.1.written,
            Some(WriteStamp {
                at_ms: 5,
                writer: 0xff
            })
        );
        for bad in [
            "",
            "x",
            "5",
            "5 ",
            " 00000000000000ff",
            "5 00000000000000f",
            "5 00000000000000fff",
            "5 00000000000000fg",
            "-5 00000000000000ff",
            "+5 00000000000000ff",
            "18446744073709551616 00000000000000ff",
            "999999999999999999999 00000000000000ff",
            "5  00000000000000ff",
            "5 00000000000000ff ",
            "--",
        ] {
            assert_eq!(parse_entry(&with_stamp(bad), &key), None, "{bad:?}");
        }
    }

    fn tombstone(target: PurgeTarget, ended_ms: Option<u64>) -> Tombstone {
        Tombstone {
            target,
            cutoff: PurgeCutoff {
                start_ms: 1_762_000_000_000,
                writer: 0xfeed,
            },
            ended_ms,
        }
    }

    #[test]
    fn tombstones_round_trip_and_authenticate() {
        let key = test_entry_key();
        for t in [
            tombstone(PurgeTarget::Key("300x200/http://x/a.jpg".into()), None),
            tombstone(PurgeTarget::Group("hero".into()), Some(1_762_000_001_000)),
            tombstone(PurgeTarget::Source("file:///a b\n.jpg".into()), None),
            tombstone(PurgeTarget::Group(String::new()), Some(0)),
        ] {
            let data = serialize_tombstone(&t, &key);
            assert_eq!(parse_tombstone(&data, &key), Some(t.clone()));
            // Under another secret it is a stranger's record.
            assert_eq!(
                parse_tombstone(&data, &crate::url::entry_key(b"other")),
                None
            );
            // Any changed byte fails the MAC or the framing.
            for i in 0..data.len() {
                let mut tampered = data.clone();
                tampered[i] ^= 0x01;
                assert_eq!(parse_tombstone(&tampered, &key), None, "byte {i}");
            }
            // Truncated or extended, it is not a record.
            assert_eq!(parse_tombstone(&data[..data.len() - 1], &key), None);
            let mut longer = data.clone();
            longer.push(b'x');
            assert_eq!(parse_tombstone(&longer, &key), None);
        }
    }

    #[test]
    fn entries_and_tombstones_never_stand_in_for_each_other() {
        // Hostile input: the bucket holds both kinds of
        // record under one secret, and a planted or moved
        // object must not parse as the other kind.
        let key = test_entry_key();
        let entry = serialize_entry("k", &image(b"x"), &key);
        assert_eq!(parse_tombstone(&entry, &key), None);
        let record = serialize_tombstone(&tombstone(PurgeTarget::Group("g".into()), None), &key);
        assert_eq!(parse_entry(&record, &key), None);
        assert_eq!(parse_entry_header(&record, record.len()), None);
        // Well-MACed but malformed bodies: an unknown kind,
        // an unstamped start, a bad end, a lying length.
        let signed = |body: &str| {
            let mac = entry_mac(&tombstone_mac_key(&key), body.as_bytes());
            let mac_hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
            format!("{TOMBSTONE_MAGIC}\n{mac_hex}\n{body}").into_bytes()
        };
        assert!(parse_tombstone(&signed("group\n5 000000000000feed\n-\n1\ng"), &key).is_some());
        for body in [
            "everything\n5 000000000000feed\n-\n1\ng",
            "group\n-\n-\n1\ng",
            "group\n5 000000000000feed\nsoon\n1\ng",
            "group\n5 000000000000feed\n-5\n1\ng",
            "group\n5 000000000000feed\n-\n2\ng",
            "group\n5 000000000000feed\n-\n0\ng",
            "group\n5 000000000000feed\n-\n18446744073709551615\ng",
            "group\n5 000000000000feed\n-\n99999999999999999999999\ng",
            "group\n5 000000000000feed\n-\n\ng",
            "group\n5 000000000000feed\n-",
        ] {
            assert_eq!(parse_tombstone(&signed(body), &key), None, "{body:?}");
        }
        // Oversized objects are refused before parsing.
        let huge = PurgeTarget::Group("g".repeat(MAX_TOMBSTONE_BYTES));
        let big = serialize_tombstone(&tombstone(huge, None), &key);
        assert_eq!(parse_tombstone(&big, &key), None);
    }

    #[test]
    fn eviction_respects_the_byte_budget() {
        let dir = temp_dir("evict");
        let payload = vec![7u8; 100];
        let file = serialize_entry("a", &image(&payload), &test_entry_key()).len();
        let one = entry_charge("a", file, &[]);
        let budget = 2 * one + one / 2;
        let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        store.put("a", &image(&payload));
        store.put("b", &image(&payload));
        store.put("c", &image(&payload));
        // The budget fits two entries, so the third store
        // evicted the least recently used.
        assert!(store.get("c").is_some());
        let (_, total) = store.stats();
        assert!(total <= budget, "total {total}");
        assert!(store.get("a").is_none(), "the LRU entry went first");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_index_record_of_each_entry_counts_against_the_budget() {
        // Tiny files under 4 KiB keys: the index holds each key
        // twice in memory. Charging only the files let the index
        // grow to about twice the budget beside them.
        let dir = temp_dir("index-charge");
        let budget = 64 * 1024;
        let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        for i in 0..100 {
            let key = format!("{i:04}/{}", "k".repeat(4091));
            store.put(&key, &image(&[7u8; 70]));
        }
        let (entries, total) = store.stats();
        assert!(total <= budget, "total {total}");
        let on_disk: usize = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap().len() as usize)
            .sum();
        let index_keys = entries * 2 * 4096;
        assert!(
            on_disk + index_keys <= budget,
            "{entries} entries: {on_disk} bytes on disk and {index_keys} bytes of index keys under a {budget}-byte budget"
        );
        // A reopened store charges the scanned entries the same.
        drop(store);
        let reopened = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        assert_eq!(reopened.stats(), (entries, total));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_entry_whose_index_record_does_not_fit_the_budget_is_not_stored() {
        // The file fits the budget; the file and the index
        // record the store keeps for it in memory do not.
        let dir = temp_dir("record-too-big");
        let key = "k".repeat(4096);
        let payload = [7u8; 70];
        let file = serialize_entry(&key, &image(&payload), &test_entry_key()).len();
        let charge = entry_charge(&key, file, &[]);
        let budget = file + 100;
        assert!(charge > budget);
        let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        store.put("small", &image(b"1"));
        let before = store.stats();
        store.put(&key, &image(&payload));
        assert_eq!(store.get(&key), None);
        assert!(
            !store.file_for(&key_stem(&key)).exists(),
            "no file was written"
        );
        assert_eq!(store.stats(), before, "nothing was evicted to make room");
        assert!(store.get("small").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_boot_scan_deletes_the_oldest_entries_until_the_rest_fit_the_budget() {
        // A directory filled under a larger budget, or by a
        // build that charged each entry its file alone, is
        // trimmed at open the way `put` would trim it, so the
        // budget holds before the first store.
        let dir = temp_dir("boot-trim");
        let keys: Vec<String> = (0..10)
            .map(|i| format!("{i:02}/{}", "k".repeat(500)))
            .collect();
        let payload = [7u8; 100];
        // Charged more than the reopened budget on its own,
        // though its file fits it.
        let huge = "h".repeat(4096);
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            for key in &keys {
                store.put(key, &image(&payload));
            }
            store.put(&huge, &image(&payload));
        }
        // Distinct modification times, oldest first, the order
        // the scan rebuilds recency from. The huge entry is the
        // newest, so trimming the oldest first would not reach it.
        for (i, key) in keys.iter().chain([&huge]).enumerate() {
            let when = std::time::SystemTime::UNIX_EPOCH
                + std::time::Duration::from_secs(1_000_000 + i as u64);
            std::fs::File::options()
                .write(true)
                .open(dir.join(format!("{}.bin", key_stem(key))))
                .unwrap()
                .set_modified(when)
                .unwrap();
        }
        let file = serialize_entry(&keys[0], &image(&payload), &test_entry_key()).len();
        let one = entry_charge(&keys[0], file, &[]);
        let budget = 3 * one + one / 2;
        let huge_file = serialize_entry(&huge, &image(&payload), &test_entry_key()).len();
        assert!(huge_file <= budget && entry_charge(&huge, huge_file, &[]) > budget);
        let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        let (entries, total) = store.stats();
        assert!(
            total <= budget,
            "{entries} entries charged {total} over a {budget}-byte budget"
        );
        assert_eq!(entries, 3);
        for key in &keys[..7] {
            assert!(store.get(key).is_none(), "an older entry was deleted");
            assert!(!dir.join(format!("{}.bin", key_stem(key))).exists());
        }
        for key in &keys[7..] {
            assert!(
                store.get(key).is_some(),
                "the newest entries that fit are kept"
            );
        }
        assert!(store.get(&huge).is_none());
        assert!(!dir.join(format!("{}.bin", key_stem(&huge))).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_file_is_a_miss_and_is_removed() {
        let dir = temp_dir("corrupt");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("k", &image(b"payload"));
        let file = store.file_for(&key_stem("k"));
        let data = std::fs::read(&file).unwrap();
        std::fs::write(&file, &data[..data.len() / 2]).unwrap();
        assert_eq!(store.get("k"), None);
        assert!(!file.exists(), "the corrupt file was cleaned up");
        // A payload-only truncation (header intact) is caught
        // by the header's payload length too.
        store.put("k", &image(b"payload"));
        let data = std::fs::read(&file).unwrap();
        std::fs::write(&file, &data[..data.len() - 3]).unwrap();
        assert_eq!(store.get("k"), None);
        // And the slot simply works again on the next store.
        store.put("k", &image(b"payload"));
        assert!(store.get("k").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn groups_round_trip_and_group_purge_survives_reopen() {
        let dir = temp_dir("groups");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("a", &grouped_image(b"1", &["thumb"]));
            store.put("b", &grouped_image(b"2", &["thumb", "hero"]));
            store.put("c", &grouped_image(b"3", &["hero"]));
            assert_eq!(store.get("b").unwrap().groups, ["thumb", "hero"]);
        }
        // Reopen: the scan re-indexes groups from the file
        // headers, so group purge works across restarts.
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(store.purge_group("thumb"), 2);
        assert!(store.get("a").is_none());
        assert!(store.get("b").is_none());
        assert!(store.get("c").is_some());
        assert_eq!(store.purge_group("thumb"), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stored entry with the given raw header count and
    /// length fields, for parser hardening tests. The MAC
    /// is computed correctly over the crafted body, so the
    /// field checks are what reject these, never the MAC.
    fn crafted_entry(group_count: &str, key_len: &str, payload_len: &str, tail: &[u8]) -> Vec<u8> {
        let mut body = format!(
            "png\n-\n{group_count}\n{key_len}\n{payload_len}\n\"00000000000000000000000000000000\"\n-\n"
        )
        .into_bytes();
        body.extend_from_slice(tail);
        let mac = entry_mac(&test_entry_key(), &body);
        let mac_hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        let mut v = format!("{MAGIC}\n{mac_hex}\n").into_bytes();
        v.extend_from_slice(&body);
        v
    }

    #[test]
    fn parser_rejects_counts_and_lengths_it_cannot_trust() {
        // Control: a normal entry with groups parses.
        let good = serialize_entry("k", &grouped_image(b"x", &["a", "b"]), &test_entry_key());
        let (key, parsed) = parse_entry(&good, &test_entry_key()).unwrap();
        assert_eq!(key, "k");
        assert_eq!(parsed.groups, ["a", "b"]);
        // A group count far beyond the bytes present is
        // rejected before any allocation is sized by it
        // (R2-13b: a tiny header used to force a multi-GB
        // with_capacity, or a capacity-overflow panic).
        let huge = usize::MAX.to_string();
        let key = test_entry_key();
        assert_eq!(
            parse_entry(&crafted_entry(&huge, "0", "0", b""), &key),
            None
        );
        assert_eq!(
            parse_entry(&crafted_entry("1000000", "1", "0", b"k\n"), &key),
            None
        );
        // Length fields at usize::MAX must fail the parse,
        // not wrap the offset arithmetic: key length,
        // payload length, and a group's length alike.
        assert_eq!(
            parse_entry(&crafted_entry("0", &huge, "0", b""), &key),
            None
        );
        assert_eq!(
            parse_entry(&crafted_entry("0", "1", &huge, b"k\n"), &key),
            None
        );
        let tail = format!("k\n{huge}\n");
        assert_eq!(
            parse_entry(&crafted_entry("1", "1", "0", tail.as_bytes()), &key),
            None
        );
        // Truncated headers and a foreign magic are misses.
        assert_eq!(parse_entry(b"CRABDISK4\n", &key), None);
        assert_eq!(parse_entry(b"CRABDISK4\npng\n", &key), None);
        // The previous format version is a miss too: its
        // files carry no MAC line, and the upgrade behavior
        // is delete on sight.
        assert_eq!(parse_entry(b"CRABDISK3\npng\n-\n0\n1\n1\nk\nx", &key), None);
        let mut foreign = serialize_entry("k", &image(b"x"), &test_entry_key());
        foreign[..9].copy_from_slice(b"CRABDISK2");
        assert_eq!(parse_entry(&foreign, &test_entry_key()), None);
    }

    #[test]
    fn boot_scan_skips_poisoned_files_and_serves_valid_entries() {
        let dir = temp_dir("poison");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("good", &image(b"payload"));
        }
        let plant = |key: &str, data: &[u8]| {
            let path = dir.join(format!("{}.bin", key_stem(key)));
            std::fs::write(&path, data).unwrap();
            path
        };
        // A header claiming a huge group count, a truncated
        // entry, and an entry larger than the whole budget
        // (internally consistent, but `put` never writes
        // one, so its size alone marks it as foreign).
        let p1 = plant(
            "evil1",
            &crafted_entry(&usize::MAX.to_string(), "0", "0", b""),
        );
        let p2 = plant("evil2", b"CRABDISK4\npng\n-");
        let big = serialize_entry("evil3", &image(&vec![7u8; 1_100_000]), &test_entry_key());
        assert!(big.len() > 1_000_000);
        let p3 = plant("evil3", &big);
        // The scan succeeds, drops all three, and still
        // serves the valid entry.
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(store.stats().0, 1);
        assert!(store.get("good").is_some());
        assert!(!p1.exists() && !p2.exists() && !p3.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_entry_grown_past_the_budget_is_a_miss_on_get() {
        let dir = temp_dir("grown");
        let store = DiskStore::open(&dir, 100_000, test_entry_key()).unwrap();
        store.put("k", &image(b"small"));
        assert!(store.get("k").is_some());
        // Replace the file with a larger, internally
        // consistent entry than the budget allows, as a
        // hostile cache writer could: the read refuses it
        // instead of buffering and serving it (R2-19).
        let big = serialize_entry("k", &image(&vec![7u8; 200_000]), &test_entry_key());
        let file = store.file_for(&key_stem("k"));
        std::fs::write(&file, &big).unwrap();
        assert_eq!(store.get("k"), None);
        assert!(!file.exists(), "the oversized file was cleaned up");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_removes_the_base_and_its_variants_only() {
        let dir = temp_dir("purge");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("300x200/http://x/a.png", &image(b"1"));
        store.put("300x200/http://x/a.png#avif", &image(b"2"));
        store.put("300x200/http://x/a.png#avif#dpr2", &image(b"3"));
        store.put("300x200/http://x/a.png2", &image(b"4"));
        assert_eq!(store.purge_matching("300x200/http://x/a.png"), 3);
        assert!(store.get("300x200/http://x/a.png").is_none());
        assert!(store.get("300x200/http://x/a.png#avif").is_none());
        assert!(store.get("300x200/http://x/a.png2").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_entries_fail_the_mac_and_read_as_misses() {
        let key = test_entry_key();
        let good = serialize_entry("k", &grouped_image(b"payload", &["hero"]), &key);
        assert!(parse_entry(&good, &key).is_some());
        // A flipped payload byte: the entry is internally
        // consistent in every field, but it is not what the
        // key holder wrote (R2-13 part a).
        let mut tampered = good.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;
        assert_eq!(parse_entry(&tampered, &key), None);
        // A flipped header byte (the stored key itself).
        let mut tampered = good.clone();
        let pos = tampered.windows(1).position(|w| w == b"k").unwrap();
        tampered[pos] = b'X';
        assert_eq!(parse_entry(&tampered, &key), None);
        // A forged MAC line over an attacker's own body.
        let mut forged = good.clone();
        forged[10] = if forged[10] == b'a' { b'b' } else { b'a' };
        assert_eq!(parse_entry(&forged, &key), None);
        // An entry written under one secret does not verify
        // under another (a replica with the wrong secret, or
        // a rotation that has closed): a miss, never served.
        let other = crate::url::entry_key(b"a different secret entirely");
        assert_eq!(parse_entry(&good, &other), None);
        // End to end through the store: a file tampered on
        // disk after indexing is a miss and is cleaned up,
        // and a file planted before boot never indexes.
        let dir = temp_dir("tampered");
        let store = DiskStore::open(&dir, 1_000_000, key.clone()).unwrap();
        store.put("k", &image(b"payload"));
        let file = store.file_for(&key_stem("k"));
        let mut data = std::fs::read(&file).unwrap();
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        std::fs::write(&file, &data).unwrap();
        assert_eq!(store.get("k"), None);
        assert!(!file.exists());
        let planted = dir.join(format!("{}.bin", key_stem("planted")));
        let mut forged = serialize_entry("planted", &image(b"forged"), &key);
        let last = forged.len() - 1;
        forged[last] ^= 0xFF;
        std::fs::write(&planted, &forged).unwrap();
        let store = DiskStore::open(&dir, 1_000_000, key).unwrap();
        assert_eq!(store.stats().0, 0, "the planted file never indexed");
        assert!(!planted.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn learned_groups_union_provenance_and_persist() {
        let dir = temp_dir("merge");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("k", &grouped_image(b"x", &["thumb"]));
            // A later spelling brings a second group: the
            // stored set grows to the union (R2-15), and a
            // repeat learn changes nothing.
            let grew = |learned| matches!(learned, Learned::Indexed { grew: true, .. });
            assert!(grew(store.learn_groups(
                "k",
                &["hero".to_string()],
                Bucket::Unknown
            )));
            assert!(!grew(store.learn_groups(
                "k",
                &["thumb".to_string()],
                Bucket::Unknown
            )));
            assert_eq!(store.get("k").unwrap().groups, ["hero", "thumb"]);
            assert_eq!(
                store.learn_groups("missing", &["hero".to_string()], Bucket::Unknown),
                Learned::Absent
            );
            // The drain writes the union into the file.
            store.drain_now();
        }
        // The union is in the file, so it survives a
        // reopen and either group purges the entry.
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(store.purge_group("hero"), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_source_removes_every_derivative_of_the_source() {
        let dir = temp_dir("purgesource");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("300x200/file:///heroes/cat.jpg", &image(b"1"));
        store.put("300x200/file:///heroes/cat.jpg#cfgab12#avif", &image(b"2"));
        store.put("600x400/file:///heroes/cat.jpg", &image(b"3"));
        store.put("300x200/file:///heroes/cat.jpg2", &image(b"4"));
        store.put("300x200/file:///heroes/dog.jpg", &image(b"5"));
        assert_eq!(store.purge_source("file:///heroes/cat.jpg"), 3);
        assert!(store.get("300x200/file:///heroes/cat.jpg").is_none());
        assert!(store.get("600x400/file:///heroes/cat.jpg").is_none());
        // A different source that merely shares a prefix,
        // and an unrelated source, are untouched.
        assert!(store.get("300x200/file:///heroes/cat.jpg2").is_some());
        assert!(store.get("300x200/file:///heroes/dog.jpg").is_some());
        assert_eq!(store.purge_source("file:///heroes/cat.jpg"), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Replaces whatever is at `path` with a FIFO.
    #[cfg(unix)]
    fn make_fifo(path: &Path) {
        let _ = std::fs::remove_file(path);
        let status = std::process::Command::new("mkfifo")
            .arg(path)
            .status()
            .expect("run mkfifo");
        assert!(status.success(), "mkfifo {}: {status}", path.display());
    }

    /// Runs `f` on a thread of its own and returns its result,
    /// failing the test if `f` is still running after a while
    /// (issue #545). A plain open of a FIFO for reading waits
    /// until a writer appears, so a store that opens the FIFO
    /// at `fifo` that way never returns. A read of a FIFO
    /// whose `writer` stays open waits for its data, so a
    /// store that reads one never returns either. Past the
    /// limit, this closes `writer` and opens the FIFO's write
    /// end until the thread finishes, so the test fails naming
    /// `what` instead of hanging.
    #[cfg(unix)]
    fn unparked<T: Send>(
        fifo: &Path,
        mut writer: Option<rustix::fd::OwnedFd>,
        what: &str,
        f: impl FnOnce() -> T + Send,
    ) -> T {
        use std::sync::mpsc::RecvTimeoutError;
        const LIMIT: std::time::Duration = std::time::Duration::from_secs(10);
        std::thread::scope(|scope| {
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let worker = scope.spawn(move || {
                let out = f();
                let _ = done_tx.send(());
                out
            });
            if let Err(RecvTimeoutError::Timeout) = done_rx.recv_timeout(LIMIT) {
                // With its only writer gone, a read of the FIFO
                // sees the end of file.
                drop(writer.take());
                while !worker.is_finished() {
                    // Opening the write end lets a reader waiting
                    // in open(2) go, and closing it is the end of
                    // file that reader sees. With no reader the
                    // open fails at once (ENXIO), so this never
                    // waits itself.
                    use rustix::fs::{Mode, OFlags};
                    drop(rustix::fs::open(
                        fifo,
                        OFlags::WRONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
                        Mode::empty(),
                    ));
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                panic!("{what} was still blocked on the FIFO after {LIMIT:?}");
            }
            worker.join().unwrap()
        })
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_at_an_entry_path_is_a_miss_and_is_removed() {
        // Issue #545: a lookup must not open a FIFO that took
        // an entry's place and wait for a writer. It is a
        // corrupt entry: a miss, never served, and deleted.
        let dir = temp_dir("fifo-get");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("k", &image(b"payload"));
        let path = store.file_for(&key_stem("k"));
        make_fifo(&path);
        let got = unparked(&path, None, "a lookup", || store.get("k"));
        assert!(got.is_none(), "a FIFO is never served");
        assert!(!store.contains("k"), "the key leaves the index");
        assert!(
            std::fs::symlink_metadata(&path).is_err(),
            "the FIFO is deleted like a corrupt entry"
        );
        // The key stores and serves again.
        store.put("k", &image(b"payload"));
        assert_eq!(store.get("k").unwrap().bytes.as_ref(), b"payload");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_boot_scan_skips_a_fifo_at_an_entry_path_unopened() {
        // Issue #545: a FIFO named like an entry held the boot
        // scan in open(2), so the server never listened. The
        // scan skips anything that is not a regular file
        // without opening it, and leaves it in place.
        let dir = temp_dir("fifo-scan");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("good", &image(b"payload"));
        }
        let fifo = dir.join(format!("{}.bin", key_stem("planted")));
        make_fifo(&fifo);
        let store = unparked(&fifo, None, "the boot scan", || {
            DiskStore::open(&dir, 1_000_000, test_entry_key())
        })
        .unwrap();
        assert_eq!(store.stats().0, 1, "only the real entry is indexed");
        assert_eq!(store.get("good").unwrap().bytes.as_ref(), b"payload");
        let meta = std::fs::symlink_metadata(&fifo).expect("the FIFO is left in place");
        assert!(std::os::unix::fs::FileTypeExt::is_fifo(&meta.file_type()));
        // A write of that key replaces the FIFO with an entry.
        store.put("planted", &image(b"fresh"));
        assert_eq!(store.get("planted").unwrap().bytes.as_ref(), b"fresh");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_with_a_writer_at_an_entry_path_is_a_miss() {
        // Issue #545: while a writer holds the FIFO open, the
        // open does not wait but a read does, for data that
        // may never come. The type check on the open file is
        // what keeps a lookup from reading it.
        use rustix::fs::{Mode, OFlags};
        let dir = temp_dir("fifo-writer-get");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("k", &image(b"payload"));
        let path = store.file_for(&key_stem("k"));
        make_fifo(&path);
        // A write end opens without waiting only while the
        // FIFO has a reader, so a reader opens first and goes
        // once the writer is in.
        let reader = rustix::fs::open(
            &path,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .expect("open the FIFO's read end");
        let writer = rustix::fs::open(
            &path,
            OFlags::WRONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .expect("open the FIFO's write end");
        drop(reader);
        let got = unparked(&path, Some(writer), "a lookup", || store.get("k"));
        assert!(got.is_none(), "a FIFO is never served");
        assert!(!store.contains("k"), "the key leaves the index");
        assert!(
            std::fs::symlink_metadata(&path).is_err(),
            "the FIFO is deleted like a corrupt entry"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_at_an_entry_path_is_a_miss_and_is_removed() {
        // Issue #545: the boot scan skips a symlink, and a
        // lookup does not follow one either, even to a valid
        // entry. It is a corrupt entry: a miss, never served,
        // and the link is deleted while its target stays.
        let dir = temp_dir("symlink-get");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("k", &image(b"payload"));
        let path = store.file_for(&key_stem("k"));
        let target = dir.join("moved-entry");
        std::fs::rename(&path, &target).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(store.get("k").is_none(), "a symlink is never followed");
        assert!(!store.contains("k"), "the key leaves the index");
        assert!(
            std::fs::symlink_metadata(&path).is_err(),
            "the link is deleted like a corrupt entry"
        );
        assert!(target.is_file(), "the link's target is left alone");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_boot_scan_leaves_a_symlink_or_directory_at_an_entry_path_unindexed() {
        // Issue #545: the scan skips anything that is not a
        // regular file, not only a FIFO. A symlink, even to a
        // valid entry, is not followed, indexed or deleted, and
        // neither is a directory. Only the symlink pins the
        // skip: a directory that reached the read would be
        // refused there, and `remove_file` cannot delete it.
        let dir = temp_dir("symlink-dir-scan");
        let (link, target) = {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("good", &image(b"payload"));
            store.put("linked", &image(b"payload"));
            let link = store.file_for(&key_stem("linked"));
            let target = dir.join("moved-entry");
            std::fs::rename(&link, &target).unwrap();
            std::os::unix::fs::symlink(&target, &link).unwrap();
            (link, target)
        };
        let planted = dir.join(format!("{}.bin", key_stem("planted")));
        std::fs::create_dir(&planted).unwrap();
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert!(!store.contains("linked"), "a symlink is never indexed");
        let meta = std::fs::symlink_metadata(&link).expect("the symlink is left in place");
        assert!(meta.file_type().is_symlink());
        assert!(target.is_file(), "the link's target is left alone");
        assert!(planted.is_dir(), "the directory is left in place");
        assert_eq!(store.stats().0, 1, "only the real entry is indexed");
        assert_eq!(store.get("good").unwrap().bytes.as_ref(), b"payload");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Issue #572: the index as the groups' authority, the
    // claims, and the background drain.

    fn groups(names: &[&str]) -> Vec<String> {
        names.iter().map(|g| g.to_string()).collect()
    }

    /// The groups in `key`'s entry file, as the boot scan
    /// reads them.
    fn file_groups(store: &DiskStore, key: &str) -> Vec<String> {
        let data = std::fs::read(store.file_for(&key_stem(key))).unwrap();
        parse_entry_header(&data, data.len()).unwrap().groups
    }

    fn grew(learned: Learned) -> bool {
        matches!(learned, Learned::Indexed { grew: true, .. })
    }

    /// A hook that holds every operation of `kind` until the
    /// returned sender sends or drops, and reports each one it
    /// holds on the returned receiver.
    fn hold_kind(
        kind: OpKind,
    ) -> (
        TestHold,
        std::sync::mpsc::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let held_tx = Mutex::new(held_tx);
        let release_rx = Mutex::new(release_rx);
        let hook: TestHold = Arc::new(move |op, _| {
            if op == kind {
                let _ = held_tx.lock().unwrap().send(());
                let _ = release_rx.lock().unwrap().recv();
            }
            Ok(())
        });
        (hook, held_rx, release_tx)
    }

    #[test]
    fn learn_groups_records_in_the_index_and_get_unions_while_the_file_is_behind() {
        let dir = temp_dir("learn");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("k", &grouped_image(b"x", &["a"]));
        assert!(grew(store.learn_groups(
            "k",
            &groups(&["c"]),
            Bucket::Unknown
        )));
        assert_eq!(store.get("k").unwrap().groups, ["a", "c"]);
        assert_eq!(file_groups(&store, "k"), ["a"], "no file I/O at learn");
        assert_eq!(store.backlog().behind, 1);
        store.drain_now();
        assert_eq!(file_groups(&store, "k"), ["a", "c"]);
        assert_eq!(store.backlog(), Backlog::default());
        assert!(!grew(store.learn_groups(
            "k",
            &groups(&["c"]),
            Bucket::Unknown
        )));
        assert_eq!(
            store.learn_groups("missing", &groups(&["c"]), Bucket::Unknown),
            Learned::Absent
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_keeps_the_file_group_order_when_up_to_date() {
        let dir = temp_dir("order");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("k", &grouped_image(b"x", &["thumb", "hero"]));
        assert!(!grew(store.learn_groups(
            "k",
            &groups(&["hero"]),
            Bucket::Unknown
        )));
        assert_eq!(store.get("k").unwrap().groups, ["thumb", "hero"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_for_an_indexed_key_learns_and_never_replaces() {
        let dir = temp_dir("held");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"x", &["a"]));
        let etag = store.get("k").unwrap().etag;
        let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = writes.clone();
        store.set_hold(Some(Arc::new(move |kind, _| {
            if kind == OpKind::Write {
                counted.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })));
        let plan = store.begin_write("k", &groups(&["c"]), Bucket::Unknown, true);
        assert!(matches!(
            plan,
            WritePlan::Held(Learned::Indexed { grew: true, .. })
        ));
        assert_eq!(writes.load(Ordering::SeqCst), 0);
        let held = store.get("k").unwrap();
        assert_eq!((held.etag, held.groups), (etag, groups(&["a", "c"])));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_in_flight_collects_groups_learned_meanwhile() {
        let dir = temp_dir("in-flight");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let WritePlan::Claimed(claim) =
            store.begin_write("k", &groups(&["c"]), Bucket::Unknown, true)
        else {
            panic!("the key was not claimed");
        };
        assert_eq!(
            store.learn_groups("k", &groups(&["d"]), Bucket::Unknown),
            Learned::InFlight
        );
        assert_eq!(store.known_groups("k").unwrap(), ["c", "d"]);
        // A group learned while the file is written lands too.
        let (hook, held, release) = hold_kind(OpKind::Write);
        store.set_hold(Some(hook));
        let writer = {
            let store = store.clone();
            std::thread::spawn(move || store.land(claim, &grouped_image(b"x", &["c"])))
        };
        held.recv().unwrap();
        store.learn_groups("k", &groups(&["e"]), Bucket::Unknown);
        drop(release);
        let landed = writer.join().unwrap().unwrap();
        assert_eq!(landed.groups, ["c", "d", "e"]);
        store.set_hold(None);
        assert_eq!(file_groups(&store, "k"), ["c", "d"]);
        store.drain_now();
        assert_eq!(file_groups(&store, "k"), ["c", "d", "e"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dropped_write_claim_unregisters() {
        let dir = temp_dir("dropped-claim");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let plan = store.begin_write("k", &groups(&["c"]), Bucket::Unknown, true);
        assert!(matches!(plan, WritePlan::Claimed(_)));
        drop(plan);
        assert_eq!(store.known_groups("k"), None);
        let plan = store.begin_write("k", &[], Bucket::Unknown, true);
        assert!(matches!(plan, WritePlan::Claimed(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_write_of_a_key_joins_the_first() {
        let dir = temp_dir("join");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let WritePlan::Claimed(claim) =
            store.begin_write("k", &groups(&["c"]), Bucket::Unknown, true)
        else {
            panic!("the key was not claimed");
        };
        assert!(matches!(
            store.begin_write("k", &groups(&["d"]), Bucket::Unknown, true),
            WritePlan::Joined
        ));
        let landed = store.land(claim, &grouped_image(b"x", &["c"])).unwrap();
        assert_eq!(landed.groups, ["c", "d"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_in_flight_when_its_key_is_purged_never_lands() {
        // A purge that covers a write in flight dooms it: the
        // write began before the purge, and its copy must not
        // stand once the purge's tombstones are gone.
        let dir = temp_dir("doomed");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        for (key, target) in [
            ("k1", PurgeTarget::Group("g".into())),
            ("k2", PurgeTarget::Key("k2".into())),
        ] {
            let WritePlan::Claimed(claim) =
                store.begin_write(key, &groups(&["g"]), Bucket::Unknown, true)
            else {
                panic!("the key was not claimed");
            };
            assert_eq!(store.purge(&target).0, 0);
            assert_eq!(store.land(claim, &grouped_image(b"x", &["g"])), None);
            assert!(!store.contains(key));
            assert!(!store.file_for(&key_stem(key)).exists());
            // The key is free for the next write.
            assert!(matches!(
                store.begin_write(key, &[], Bucket::Unknown, true),
                WritePlan::Claimed(_)
            ));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_doomed_write_teaches_no_known_groups() {
        // A copy written below takes the groups this process
        // knows. Those of a write a purge doomed are under the
        // purge, and would make that copy one the purge drops.
        let dir = temp_dir("doomed-known");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let WritePlan::Claimed(claim) =
            store.begin_write("k", &groups(&["g"]), Bucket::Unknown, true)
        else {
            panic!("the key was not claimed");
        };
        assert_eq!(store.known_groups("k"), Some(groups(&["g"])));
        store.purge(&PurgeTarget::Group("g".into()));
        assert_eq!(store.known_groups("k"), None);
        drop(claim);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn purges_remove_records_at_once_and_queue_the_files() {
        let dir = temp_dir("purge-queue");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        store.put("a", &image(b"1"));
        store.put("a#avif", &image(b"2"));
        store.put("b", &image(b"3"));
        let (removed, seqs) = store.purge(&PurgeTarget::Key("a".into()));
        assert_eq!((removed, seqs.end - seqs.start), (2, 2));
        assert!(!store.contains("a") && !store.contains("a#avif"));
        assert!(
            store.file_for(&key_stem("a")).exists(),
            "no file I/O at purge"
        );
        assert!(store.deletions_pending(&seqs));
        assert_eq!(store.backlog().deletions, 2);
        store.drain_now();
        assert!(!store.file_for(&key_stem("a")).exists());
        assert!(!store.file_for(&key_stem("a#avif")).exists());
        assert!(!store.deletions_pending(&seqs));
        assert!(store.get("b").is_some());
        // A purge that matched nothing queued nothing.
        let (removed, seqs) = store.purge(&PurgeTarget::Group("none".into()));
        assert_eq!(removed, 0);
        assert!(seqs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn learned_groups_are_charged_and_the_next_landing_evicts_to_fit() {
        let dir = temp_dir("learn-charge");
        let payload = vec![7u8; 100];
        let file = serialize_entry("a", &image(&payload), &test_entry_key()).len();
        let budget = 2 * entry_charge("a", file, &[]) + 200;
        let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        store.put("a", &image(&payload));
        store.put("b", &image(&payload));
        let before = store.stats().1;
        let group = "g".repeat(300);
        store.learn_groups("a", std::slice::from_ref(&group), Bucket::Unknown);
        assert!(
            store.stats().1 > before,
            "the learned group was not charged"
        );
        assert!(store.stats().1 > budget, "the learn itself evicts nothing");
        store.put("c", &image(&payload));
        assert!(store.stats().1 <= budget);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_evicted_key_is_unlinked_under_a_claim() {
        let dir = temp_dir("evict-claim");
        let payload = vec![7u8; 100];
        let file = serialize_entry("a", &image(&payload), &test_entry_key()).len();
        let budget = entry_charge("a", file, &[]) + 50;
        let store = Arc::new(DiskStore::open(&dir, budget, test_entry_key()).unwrap());
        store.put("a", &image(&payload));
        let (hook, held, release) = hold_kind(OpKind::Unlink);
        store.set_hold(Some(hook));
        let writer = {
            let store = store.clone();
            let payload = payload.clone();
            std::thread::spawn(move || store.put("b", &image(&payload)))
        };
        // The landing of `b` evicts `a` and deletes its file
        // under `a`'s claim: a fresh write of `a` meanwhile
        // writes nothing.
        held.recv().unwrap();
        assert!(!store.contains("a"));
        assert!(matches!(
            store.begin_write("a", &[], Bucket::Unknown, true),
            WritePlan::Busy
        ));
        drop(release);
        writer.join().unwrap();
        store.set_hold(None);
        assert!(!store.file_for(&key_stem("a")).exists());
        assert!(matches!(
            store.begin_write("a", &[], Bucket::Unknown, true),
            WritePlan::Claimed(_)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_misses_when_the_record_is_replaced_during_the_read() {
        // A read of copy X whose record a purge removes and a
        // fresh write Y replaces before the read returns: X's
        // bytes are a purged copy's, and they must not serve
        // with Y's groups, which may lack the purged group.
        let dir = temp_dir("replaced-read");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"x", &["a"]));
        store.learn_groups("k", &groups(&["g"]), Bucket::Unknown);
        let (hook, held, release) = hold_kind(OpKind::Read);
        store.set_hold(Some(hook));
        let reader = {
            let store = store.clone();
            std::thread::spawn(move || store.get_with_id("k"))
        };
        held.recv().unwrap();
        store.set_hold(None);
        assert_eq!(store.purge(&PurgeTarget::Group("g".into())).0, 1);
        store.drain_now();
        store.put("k", &grouped_image(b"y", &["b"]));
        drop(release);
        assert_eq!(reader.join().unwrap(), None);
        assert_eq!(store.get("k").unwrap().bytes.as_ref(), b"y");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stall_lines_are_rate_limited() {
        let log = StallLog::new("disk waits ran out");
        let t0 = 1_000_000;
        let first = log.note(t0, "a disk wait ran out").unwrap();
        assert_eq!(first, "result storage: a disk wait ran out");
        for i in 1..50 {
            assert_eq!(log.note(t0 + i * 100, "a disk wait ran out"), None);
        }
        let next = log.note(t0 + STALL_LOG_EVERY_MS, "a disk wait ran out");
        assert_eq!(
            next.unwrap(),
            "result storage: a disk wait ran out; 49 more disk waits ran out since the last line"
        );
        // Every stall counts, logged or not.
        let dir = temp_dir("stall-count");
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        for _ in 0..50 {
            store.note_stall(Stall::LookupRead);
        }
        assert_eq!(store.errors(), 50);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_failed_rewrite_stays_queued_and_counted_until_it_lands() {
        let dir = temp_dir("failed-rewrite");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"x", &["a"]));
        store.set_hold(Some(Arc::new(|kind, _| match kind {
            OpKind::Write => Err(std::io::Error::other("disk full")),
            _ => Ok(()),
        })));
        store.learn_groups("k", &groups(&["g"]), Bucket::Unknown);
        store.kick();
        let failed = async {
            while store.errors() == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), failed)
            .await
            .unwrap();
        let backlog = store.backlog();
        assert_eq!((backlog.behind, backlog.failing), (1, 1));
        // A shutdown does not wait for a job that keeps failing.
        let started = std::time::Instant::now();
        let left = store
            .settle(tokio::time::Instant::now() + std::time::Duration::from_secs(5))
            .await;
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(left.behind, 1);
        // Once the disk takes writes, the retry lands.
        store.set_hold(None);
        let landed = async {
            while file_groups(&store, "k") != ["a", "g"] {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(10), landed)
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.backlog() != Backlog::default() {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_panicking_drain_job_releases_its_claim_and_requeues() {
        let dir = temp_dir("panicking-job");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &image(b"x"));
        store.purge(&PurgeTarget::Key("k".into()));
        store.set_hold(Some(Arc::new(|kind, _| {
            if kind == OpKind::Unlink {
                panic!("the disk went away");
            }
            Ok(())
        })));
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.run_one_job()));
        assert!(ran.is_err());
        store.set_hold(None);
        assert_eq!(store.backlog().deletions, 1, "the deletion is not queued");
        let plan = store.begin_write("k", &[], Bucket::Unknown, true);
        assert!(matches!(plan, WritePlan::Claimed(_)), "{plan:?}");
        drop(plan);
        store.drain_now();
        assert!(!store.file_for(&key_stem("k")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_queued_deletion_never_unlinks_a_fresh_write() {
        // No runtime here, so nothing starts the deletion the
        // purge queues before the fresh write lands.
        let dir = temp_dir("fresh-over-deletion");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &image(b"old"));
        store.purge(&PurgeTarget::Key("k".into()));
        let WritePlan::Claimed(claim) = store.begin_write("k", &[], Bucket::Unknown, true) else {
            panic!("the key was not claimed");
        };
        store.land(claim, &image(b"new")).unwrap();
        store.drain_now();
        assert!(store.file_for(&key_stem("k")).exists());
        assert_eq!(store.get("k").unwrap().bytes.as_ref(), b"new");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_purge_during_a_rewrite_deletes_the_rewritten_file() {
        let dir = temp_dir("purge-rewrite");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"x", &["a"]));
        store.learn_groups("k", &groups(&["g"]), Bucket::Unknown);
        let (hook, held, release) = hold_kind(OpKind::Write);
        store.set_hold(Some(hook));
        let drain = {
            let store = store.clone();
            std::thread::spawn(move || store.run_one_job())
        };
        held.recv().unwrap();
        assert_eq!(store.purge(&PurgeTarget::Group("a".into())).0, 1);
        // The rewrite holds the file: a fresh write of the key
        // waits for nothing and writes nothing.
        assert!(matches!(
            store.begin_write("k", &[], Bucket::Unknown, true),
            WritePlan::Busy
        ));
        drop(release);
        assert!(drain.join().unwrap());
        store.set_hold(None);
        assert!(
            !store.file_for(&key_stem("k")).exists(),
            "the rewrite left its file behind a purged record"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fresh_write_during_a_deletion_is_busy() {
        let dir = temp_dir("busy-delete");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &image(b"x"));
        store.purge(&PurgeTarget::Key("k".into()));
        let (hook, held, release) = hold_kind(OpKind::Unlink);
        store.set_hold(Some(hook));
        let drain = {
            let store = store.clone();
            std::thread::spawn(move || store.run_one_job())
        };
        held.recv().unwrap();
        assert!(matches!(
            store.begin_write("k", &[], Bucket::Unknown, true),
            WritePlan::Busy
        ));
        drop(release);
        assert!(drain.join().unwrap());
        store.set_hold(None);
        assert!(matches!(
            store.begin_write("k", &[], Bucket::Unknown, true),
            WritePlan::Claimed(_)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn groups_learned_for_many_keys_during_a_stall_all_land() {
        // Issue #572, race 3: group merges queued during a
        // stall overwrote each other. Each key's file now has
        // one writer at a time, and a group learned while it
        // writes queues the key again.
        let dir = temp_dir("stall-groups");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let keys: Vec<String> = (0..15).map(|i| format!("k{i}")).collect();
        for key in &keys {
            store.put(key, &grouped_image(key.as_bytes(), &["a"]));
        }
        let (hook, _held, release) = hold_kind(OpKind::Write);
        store.set_hold(Some(hook));
        for key in &keys {
            store.learn_groups(key, &groups(&["c"]), Bucket::Unknown);
        }
        store.kick();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        for key in &keys {
            store.learn_groups(key, &groups(&["d"]), Bucket::Unknown);
        }
        store.kick();
        store.set_hold(None);
        drop(release);
        let left = store
            .settle(tokio::time::Instant::now() + std::time::Duration::from_secs(10))
            .await;
        assert_eq!(left, Backlog::default());
        drop(store);
        let reopened = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        for key in &keys {
            assert_eq!(file_groups(&reopened, key), ["a", "c", "d"], "{key}");
        }
        assert_eq!(reopened.purge(&PurgeTarget::Group("c".into())).0, 15);
        let reopened = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(reopened.purge(&PurgeTarget::Group("d".into())).0, 15);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn settle_returns_at_its_deadline_with_the_backlog_and_at_once_when_idle() {
        let dir = temp_dir("settle");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let started = std::time::Instant::now();
        let left = store
            .settle(tokio::time::Instant::now() + std::time::Duration::from_secs(5))
            .await;
        assert_eq!(left, Backlog::default());
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        store.put("k", &grouped_image(b"x", &["a"]));
        let (hook, _held, release) = hold_kind(OpKind::Write);
        store.set_hold(Some(hook));
        store.learn_groups("k", &groups(&["g"]), Bucket::Unknown);
        let started = std::time::Instant::now();
        let left = store
            .settle(tokio::time::Instant::now() + std::time::Duration::from_millis(300))
            .await;
        assert!(started.elapsed() >= std::time::Duration::from_millis(300));
        assert_eq!(left.behind, 1);
        store.set_hold(None);
        drop(release);
        let left = store
            .settle(tokio::time::Instant::now() + std::time::Duration::from_secs(10))
            .await;
        assert_eq!(left, Backlog::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_purge_waits_only_for_its_own_deletions_and_not_for_failed_ones() {
        let dir = temp_dir("own-deletions");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("a", &image(b"1"));
        store.put("b", &image(b"2"));
        let failing = store.file_for(&key_stem("a"));
        store.set_hold(Some(Arc::new(move |kind, path| {
            if kind == OpKind::Unlink && path == failing {
                return Err(std::io::Error::other("read-only"));
            }
            Ok(())
        })));
        let wait = |seqs| {
            let store = store.clone();
            async move {
                let started = std::time::Instant::now();
                let done = store
                    .wait_deletions(
                        seqs,
                        std::time::Duration::from_secs(1),
                        std::time::Duration::from_secs(30),
                    )
                    .await;
                (done, started.elapsed())
            }
        };
        let (_, a) = store.purge(&PurgeTarget::Key("a".into()));
        store.kick();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.backlog().failing == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let (_, b) = store.purge(&PurgeTarget::Key("b".into()));
        let (done, took) = wait(b).await;
        assert!(
            done && took < std::time::Duration::from_millis(900),
            "{took:?}"
        );
        assert!(!store.file_for(&key_stem("b")).exists());
        // The failed one is retried and counted, not waited for.
        let (done, took) = wait(a.clone()).await;
        assert!(
            done && took < std::time::Duration::from_millis(900),
            "{took:?}"
        );
        assert!(store.deletions_pending(&a));
        let (done, _) = wait(store.purge(&PurgeTarget::Key("none".into())).1).await;
        assert!(done);
        store.set_hold(None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Index `n` keys, `k0` and on, each in `group(i)`, with
    /// no files: for tests of what the queues cost.
    fn index_without_files(store: &DiskStore, n: usize, group: impl Fn(usize) -> String) {
        let mut guard = store.inner.lock().unwrap();
        let inner = &mut *guard;
        for i in 0..n {
            let key = format!("k{i}");
            let groups = vec![group(i)];
            let charge = entry_charge(&key, 10, &groups);
            inner.next_id += 1;
            inner.clock += 1;
            inner.stamps.insert(key.clone(), inner.clock);
            inner.total_bytes += charge;
            let record = Record {
                stem: key_stem(&key),
                file_bytes: 10,
                charge,
                groups,
                written: None,
                id: inner.next_id,
                owes_shared: false,
            };
            inner.entries.insert(key, record);
        }
    }

    #[test]
    fn a_purge_of_keys_with_queued_rewrites_takes_time_linear_in_them() {
        // Issue #572 review: a purge dropped each key's queued
        // rewrite with a scan of the whole queue, under the
        // index lock, so 20,000 keys held it for seconds.
        let dir = temp_dir("purge-queued-rewrites");
        let store = DiskStore::open(&dir, usize::MAX / 4, test_entry_key()).unwrap();
        let n = 20_000;
        index_without_files(&store, n, |_| "a".to_string());
        for i in 0..n {
            store.learn_groups(&format!("k{i}"), &groups(&["g"]), Bucket::Unknown);
        }
        assert_eq!(store.queued(), (n, 0));
        let started = std::time::Instant::now();
        let (removed, _) = store.purge(&PurgeTarget::Group("g".into()));
        let took = started.elapsed();
        assert_eq!(removed, n);
        assert_eq!(store.queued(), (0, n));
        assert!(took < std::time::Duration::from_millis(1500), "{took:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A store whose `n` purged keys each failed their deletion
    /// once, as on a disk remounted read-only, and wait out a
    /// backoff, with one drain task left waiting for them.
    async fn failing_backlog(
        tag: &str,
        n: usize,
    ) -> (crate::test_support::ScratchDir, Arc<DiskStore>) {
        let dir = temp_dir(tag);
        let store = Arc::new(DiskStore::open(&dir, usize::MAX / 4, test_entry_key()).unwrap());
        index_without_files(&store, n, |_| "g".to_string());
        store.purge(&PurgeTarget::Group("g".into()));
        store.set_hold(Some(Arc::new(|kind, _| match kind {
            OpKind::Unlink => Err(std::io::Error::from_raw_os_error(30)),
            _ => Ok(()),
        })));
        store.kick();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.backlog().failing < n || store.inner.lock().unwrap().workers > 1 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        (dir, store)
    }

    #[tokio::test]
    async fn while_every_job_waits_out_a_backoff_kicks_start_no_drain_task() {
        // Issue #572 review: jobs in a backoff stayed in the
        // queue, and each kick started a drain task that
        // scanned them all, so every disk operation paid for
        // the whole failing backlog.
        let (dir, store) = failing_backlog("backoff-kicks", 50).await;
        assert_eq!(store.inner.lock().unwrap().deletions.order.len(), 0);
        for _ in 0..100 {
            store.kick();
            assert_eq!(store.inner.lock().unwrap().workers, 1);
        }
        store.set_hold(None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn failed_background_jobs_are_logged_at_most_once_per_interval() {
        let (dir, store) = failing_backlog("job-lines", 50).await;
        assert_eq!(store.errors(), 50, "every failure counts");
        assert_eq!(
            store.job_log.since.load(Ordering::Relaxed),
            49,
            "every failure logged"
        );
        store.set_hold(None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_purge_wait_runs_out_when_its_own_deletions_stop_while_other_jobs_finish() {
        // Issue #572 review: the purge's idle bound restarted
        // whenever any background job ended, so a purge whose
        // own deletion hung waited its whole cap.
        let dir = temp_dir("own-idle");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("stuck", &image(b"s"));
        for i in 0..20 {
            store.put(&format!("other{i}"), &image(b"o"));
        }
        let stuck = store.file_for(&key_stem("stuck"));
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Mutex::new(released);
        store.set_hold(Some(Arc::new(move |kind, path| {
            if kind == OpKind::Unlink && path == stuck {
                let _ = released
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(20));
            }
            Ok(())
        })));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let teacher = {
            let (store, stop) = (store.clone(), stop.clone());
            tokio::spawn(async move {
                let mut i = 0;
                while !stop.load(Ordering::Relaxed) {
                    let key = format!("other{}", i % 20);
                    store.learn_groups(&key, &[format!("t{i}")], Bucket::Unknown);
                    store.kick();
                    i += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            })
        };
        let (_, seqs) = store.purge(&PurgeTarget::Key("stuck".into()));
        let started = std::time::Instant::now();
        let done = store
            .wait_deletions(
                seqs,
                std::time::Duration::from_millis(500),
                std::time::Duration::from_secs(10),
            )
            .await;
        let took = started.elapsed();
        stop.store(true, Ordering::Relaxed);
        teacher.await.unwrap();
        drop(release);
        store.set_hold(None);
        assert!(!done);
        assert!(took < std::time::Duration::from_secs(3), "{took:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// How long 4,000 queued deletions from 1,000 purges take to
    /// drain once a stuck disk answers, with or without a wait
    /// on each purge's deletions.
    async fn drain_after_a_stall(waiters: bool) -> std::time::Duration {
        let dir = temp_dir(&format!("drain-waiters-{waiters}"));
        let store = Arc::new(DiskStore::open(&dir, usize::MAX / 4, test_entry_key()).unwrap());
        index_without_files(&store, 4_000, |i| format!("g{}", i / 4));
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let hook = gate.clone();
        store.set_hold(Some(Arc::new(move |kind, _| {
            if kind == OpKind::Unlink {
                let (released, wake) = &*hook;
                let released = released.lock().unwrap();
                let _ = wake
                    .wait_timeout_while(released, std::time::Duration::from_secs(30), |r| !*r)
                    .unwrap();
            }
            Ok(())
        })));
        let mut waits = Vec::new();
        for g in 0..1_000 {
            let (_, seqs) = store.purge(&PurgeTarget::Group(format!("g{g}")));
            if waiters {
                let store = store.clone();
                waits.push(tokio::spawn(async move {
                    store
                        .wait_deletions(
                            seqs,
                            std::time::Duration::from_secs(60),
                            std::time::Duration::from_secs(120),
                        )
                        .await
                }));
            }
        }
        store.kick();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let started = std::time::Instant::now();
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        tokio::time::timeout(std::time::Duration::from_secs(60), async {
            while store.queued().1 > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        for wait in waits {
            assert!(wait.await.unwrap());
        }
        let took = started.elapsed();
        store.set_hold(None);
        let _ = std::fs::remove_dir_all(&dir);
        took
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn purges_waiting_on_their_deletions_do_not_slow_the_drain() {
        // Issue #572 review: each wait scanned every queued
        // deletion whenever any job ended, so 1,000 waiting
        // purges made the drain after a stall 60 times slower.
        let alone = drain_after_a_stall(false).await;
        let waited = drain_after_a_stall(true).await;
        assert!(
            waited < alone * 5 + std::time::Duration::from_secs(1),
            "{alone:?} with no waits, {waited:?} with 1,000"
        );
    }

    #[tokio::test]
    async fn work_queued_without_a_kick_starts_after_the_next_disk_operation() {
        // Each operation on the blocking pool kicks the drain
        // before it ends, so a caller that stops waiting for it
        // cannot leave work it queued unstarted.
        let dir = temp_dir("kick-after-op");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"x", &["a"]));
        store.learn_groups("k", &groups(&["g"]), Bucket::Unknown);
        assert_eq!(store.queued(), (1, 0));
        store.blocking(|store| store.contains("other")).await;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.queued() != (0, 0) {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(file_groups(&store, "k"), ["a", "g"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
