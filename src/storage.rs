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
//! from file times by the boot scan. Each entry is charged its
//! file and the record the in-memory index keeps for it, so
//! the budget bounds the index's memory as well as the files.
//!
//! The index is the authority on an entry's cache groups
//! (issue #572). A group learned on a hit goes into the index
//! at once, with no file I/O, and a background drain writes it
//! into the entry file later. Purges remove index records at
//! once and the drain deletes their files. Only one writer or
//! deleter works on an entry file at a time: it holds the
//! key's claim.
//!
//! The server opens the store with [`DiskStore::start`] (issue
//! #708): the directory check and the boot scan, which reads
//! and verifies every entry file, run on a thread of their
//! own, so a slow or hung disk does not keep the server from
//! listening. Until the scan ends, an entry it has not indexed
//! is a miss, writes land and evict as always, and the purges
//! sent meanwhile are applied to the files it indexes (see
//! [`ScanState`]).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

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
    /// When the entry was last used: its key's place in
    /// [`Inner::order`] (issue #708).
    stamp: u64,
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
    /// Being run right now, since then.
    running: Option<std::time::Instant>,
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
    /// The deletions queued outside a purge, which no purge
    /// holds its tombstones for yet (a lookup, a landing or a
    /// learn that found the copy under a purge, or an
    /// eviction), by key, with the groups of the copy each
    /// deletes, until a purge that covers the copy takes it
    /// over (see [`DiskStore::purge_waiting`]). Kept apart so
    /// a purge looks at these only, not the whole queue.
    unheld: HashMap<String, Vec<String>>,
    /// Jobs running now.
    running: usize,
    /// Jobs that failed at least once.
    failing: usize,
    /// The sequence numbers of jobs that stopped counting for
    /// the waits on them since those last heard: each job once,
    /// when it ends, goes, or first fails, with `track_ended`
    /// (deletions only).
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
                running: None,
            },
        );
        self.by_seq.insert(*seq, key.to_string());
        self.order.push_back((key.to_string(), *seq));
    }

    /// [`KeyQueue::push`] for the deletion of a copy in
    /// `groups` that no purge holds its tombstones for.
    fn push_unheld(&mut self, key: &str, stem: &str, groups: &[String], seq: &mut u64) {
        if self.jobs.contains_key(key) {
            return;
        }
        self.push(key, stem, seq);
        self.unheld.insert(key.to_string(), groups.to_vec());
    }

    /// Drop `key`'s job. Returns whether there was one.
    fn remove(&mut self, key: &str) -> bool {
        let Some(job) = self.jobs.remove(key) else {
            return false;
        };
        self.by_seq.remove(&job.seq);
        self.unheld.remove(key);
        // A failed job was counted out when it first failed.
        if self.track_ended && job.attempts == 0 {
            self.ended.push(job.seq);
        }
        if let Some(at) = job.not_before {
            self.later.remove(&(at, job.seq));
        }
        if job.running.is_some() {
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
                && job.running.is_none()
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
            job.running = Some(std::time::Instant::now());
            self.running += 1;
        }
    }

    /// End a running job: drop it, or keep it queued to run
    /// again at once.
    fn finish(&mut self, key: &str, again: bool) {
        match self.jobs.get_mut(key) {
            Some(job) if again && job.running.is_some() => {
                job.running = None;
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
            && job.running.is_some()
        {
            job.running = None;
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

    /// How many jobs in `seqs` are queued or running and have
    /// not failed. Walks the range: once per wait.
    fn count_in(&self, seqs: &Range<u64>) -> usize {
        self.by_seq
            .range(seqs.clone())
            .filter(|(_, key)| self.jobs.get(*key).is_some_and(|job| job.attempts == 0))
            .count()
    }

    /// Whether a job in `seqs` is queued or running, failed or
    /// not.
    fn any_in(&self, seqs: &Range<u64>) -> bool {
        self.by_seq.range(seqs.clone()).next().is_some()
    }

    /// Whether a job in `seqs` has been running since `since` or
    /// before. Walks the range: once per wait that runs out.
    fn running_since(&self, seqs: &Range<u64>, since: std::time::Instant) -> bool {
        self.by_seq.range(seqs.clone()).any(|(_, key)| {
            self.jobs
                .get(key)
                .and_then(|job| job.running)
                .is_some_and(|started| started <= since)
        })
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
/// copies and its file stem. The two are counted as they were
/// when the recency order was a second hash map (issue #708): a
/// key with its record, and a key with a stamp. The stamp a
/// record now carries is not added, so an entry is charged what
/// it always was and a budget holds the entries it held.
const RECORD_OVERHEAD: usize = 2
    * (size_of::<(String, Record)>() - size_of::<u64>() + size_of::<(String, u64)>())
    + 3 * ALLOCATION_OVERHEAD;

/// Where the clock of last use starts (issue #708). The boot
/// scan stamps the entries it indexes below it, from 1 up in
/// the order of their files' times, so every entry it indexes
/// is older than every entry used since the store opened, a
/// write that landed while it ran included.
const SCAN_STAMPS: u64 = 1 << 62;

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
    /// The indexed keys in the order they were last used: each
    /// under its record's stamp, a reading of `clock`, so no
    /// two are equal. Eviction takes the smallest stamp, the
    /// least recently used entry, from the front (issue #708).
    /// While the stamps sat in a hash map, every eviction read
    /// every entry's stamp to find it, under the index lock.
    order: BTreeMap<u64, String>,
    clock: u64,
    /// The sum of the indexed entries' charges.
    total_bytes: usize,
    /// Waits on one purge's deletions, by the first sequence
    /// number of its range: the range's end and the wait's
    /// count. Purge ranges never overlap.
    deletion_waits: BTreeMap<u64, Vec<(u64, Arc<DeletionCount>)>>,
    /// The boot scan's state from the store's start until the
    /// scan has indexed what it found, or has stopped (issue
    /// #708).
    scan: Option<ScanState>,
    /// The directory failed its check after the boot went on
    /// without it (issue #708): nothing is written to it.
    unusable: bool,
}

/// What the boot scan needs to know of the work done while it
/// runs (issue #708). The scan reads every entry file first,
/// without touching the index, and then indexes what it found
/// in one step under the index lock. Requests are served
/// meanwhile, so by then a file it read may be stale: a write
/// may have replaced it, a purge may cover it. These are what
/// it checks.
#[derive(Debug, Default)]
struct ScanState {
    /// The directory has been created, probed and opened.
    /// Until then nothing is written.
    checked: bool,
    /// The purges sent since the store started. The scan
    /// deletes the files it read that one covers instead of
    /// indexing them: they were written before the store
    /// started, so before the purge.
    purges: ScanPurges,
    /// The file stems of the keys a write was begun for, or
    /// refused because the directory had not answered or the
    /// scan was deleting a file at the stem. The copy a write
    /// lands replaces the file the scan may have read, and the
    /// copy it was for serves from memory, refused or not, with
    /// groups the scanned file lacks, so the scan indexes no
    /// such file: it deletes it, unless the write's copy is
    /// indexed. Kept in a filter of fixed size (see
    /// [`StemFilter`]): on a disk that hangs, the scan lasts as
    /// long as the hang, and every key rendered meanwhile is
    /// written.
    written: StemFilter,
    /// The stems of the files the scan is deleting by name,
    /// not knowing their key: files that do not parse, or name
    /// another key's entry. A write of a key with such a stem
    /// waits for none and writes nothing, so the deletion never
    /// removes a file it renamed into place.
    unlinking: HashSet<String>,
}

/// A set of file stems in a fixed amount of memory (issue
/// #708): a Bloom filter of [`StemFilter::BITS`] bits, 2 MiB,
/// allocated at the first stem. A stem is a hex SHA-256, so its
/// first four groups of eight hex digits are four independent,
/// uniform numbers, and each picks one bit. The filter never
/// misses a stem it holds. Once it holds many, it can also
/// claim one it does not: the boot scan then deletes that key's
/// file instead of indexing it, or leaves a file that does not
/// verify for the next boot to delete. Either costs a render at
/// most, never a stale copy. A million stems make about one in
/// 500 of the others look held; ten thousand, about one in 30
/// billion.
#[derive(Default)]
struct StemFilter {
    bits: Vec<u64>,
}

impl StemFilter {
    const BITS: usize = 1 << 24;

    /// The bits `stem` sets. Whatever the text, the same stem
    /// always gives the same bits, so nothing it holds is
    /// missed.
    fn bits_of(stem: &str) -> [usize; 4] {
        std::array::from_fn(|n| {
            stem.get(8 * n..8 * n + 8)
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .map_or(0, |value| value as usize % Self::BITS)
        })
    }

    fn insert(&mut self, stem: &str) {
        if self.bits.is_empty() {
            self.bits = vec![0; Self::BITS / 64];
        }
        for bit in Self::bits_of(stem) {
            self.bits[bit / 64] |= 1 << (bit % 64);
        }
    }

    fn contains(&self, stem: &str) -> bool {
        !self.bits.is_empty()
            && Self::bits_of(stem)
                .into_iter()
                .all(|bit| self.bits[bit / 64] & (1 << (bit % 64)) != 0)
    }
}

impl std::fmt::Debug for StemFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let set: u32 = self.bits.iter().map(|word| word.count_ones()).sum();
        f.debug_struct("StemFilter")
            .field("bits_set", &set)
            .finish()
    }
}

/// A purge sent while the boot scan runs (issue #708): the
/// deletions of the files the scan found under it, which the
/// purge's bucket tombstones wait for, as they wait for the
/// deletions the purge queued itself (see
/// [`DeletionWait::pending`]). A restart before the files are
/// gone indexes them again, and the tombstone, still unended,
/// keeps them misses.
#[derive(Debug, Default)]
struct ScanPurge {
    /// Set once the scan has applied the purge: the sequence
    /// numbers of the deletions it queued for it. Never set
    /// when the scan stops or fails first, so the tombstones
    /// stay in force.
    applied: OnceLock<Vec<u64>>,
}

impl ScanPurge {
    /// Whether the scan has yet to apply the purge, or a
    /// deletion it queued for it is still queued.
    fn pending(&self, deletions: &KeyQueue) -> bool {
        self.applied
            .get()
            .is_none_or(|seqs| seqs.iter().any(|seq| deletions.by_seq.contains_key(seq)))
    }
}

/// The purges sent while the boot scan runs (issue #708): one
/// [`ScanPurge`] per target, which every purge of that target
/// shares, filed by the name its kind matches on. The scan
/// judges every file it read against them under the index
/// lock, so it looks up the few names that could cover a file
/// instead of trying every target: a slow scan of a large
/// directory during many purges does not hold the lock for
/// files times targets.
#[derive(Debug, Default)]
struct ScanPurges {
    /// How many purges were sent, repeats included.
    sent: usize,
    /// One per target.
    holds: Vec<Arc<ScanPurge>>,
    /// Where each target's hold is in `holds`: a per-URL
    /// purge's by its base key, a group purge's by its group,
    /// a source purge's by its source.
    keys: HashMap<String, usize>,
    groups: HashMap<String, usize>,
    sources: HashMap<String, usize>,
}

impl ScanPurges {
    /// Record a purge of `target`, and return the hold its
    /// tombstones wait on.
    fn add(&mut self, target: &PurgeTarget) -> Arc<ScanPurge> {
        self.sent += 1;
        let (filed, name) = match target {
            PurgeTarget::Key(base) => (&mut self.keys, base),
            PurgeTarget::Group(group) => (&mut self.groups, group),
            PurgeTarget::Source(source) => (&mut self.sources, source),
        };
        let holds = &mut self.holds;
        let at = *filed.entry(name.clone()).or_insert_with(|| {
            holds.push(Arc::new(ScanPurge::default()));
            holds.len() - 1
        });
        self.holds[at].clone()
    }

    /// Where in `holds` the purges that cover a copy of `key`
    /// in `groups` are: those whose [`PurgeTarget::covers`]
    /// is true. A per-URL purge covers the key when its base is
    /// the key, or the key up to one of its `#`; a group purge
    /// when its group is one of `groups`; a source purge when
    /// its source is the key's base (the key up to its first
    /// `#`), or that base after one of its `/`.
    fn covering(&self, key: &str, groups: &[String]) -> Vec<usize> {
        let mut found = Vec::new();
        if !self.keys.is_empty() {
            let bases = key.match_indices('#').map(|(at, _)| &key[..at]);
            found.extend(
                bases
                    .chain([key])
                    .filter_map(|base| self.keys.get(base).copied()),
            );
        }
        if !self.groups.is_empty() {
            found.extend(
                groups
                    .iter()
                    .filter_map(|group| self.groups.get(group).copied()),
            );
        }
        if !self.sources.is_empty() {
            let base = key.split('#').next().unwrap_or(key);
            let tails = base.match_indices('/').map(|(at, _)| &base[at + 1..]);
            found.extend(
                tails
                    .chain([base])
                    .filter_map(|tail| self.sources.get(tail).copied()),
            );
        }
        found
    }
}

/// One wait on a purge's deletions (issue #572): how many of
/// them are still queued or running and have not failed, kept
/// as each one ends, goes or fails, so the wait never counts
/// them again, and its wake.
#[derive(Debug, Default)]
struct DeletionCount {
    left: std::sync::atomic::AtomicUsize,
    wake: tokio::sync::Notify,
}

impl Inner {
    fn new() -> Self {
        Inner {
            deletions: KeyQueue {
                track_ended: true,
                ..KeyQueue::default()
            },
            clock: SCAN_STAMPS,
            scan: Some(ScanState::default()),
            ..Inner::default()
        }
    }

    /// Count down and wake the waits whose purge had a deletion
    /// end, fail or go since they last heard (issue #572). Only
    /// those wake, each at a constant cost, so many waiting
    /// purges, or one purge of many files, cost the drain
    /// nothing.
    fn wake_deletion_waits(&mut self) {
        for seq in self.deletions.ended.drain(..) {
            if let Some((_, waits)) = self.deletion_waits.range(..=seq).next_back() {
                for (end, count) in waits {
                    if seq < *end {
                        // Only this, under the index lock, writes
                        // the count; the wait only reads it.
                        let left = count.left.load(Ordering::Relaxed);
                        count.left.store(left.saturating_sub(1), Ordering::Relaxed);
                        count.wake.notify_one();
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
    /// Zeroed when the store is dropped.
    entry_key: zeroize::Zeroizing<Vec<u8>>,
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
    /// When a background job last finished its work, on
    /// [`monotonic_ms`] (0: never): a disk that finishes jobs
    /// works, even while one purge's deletions wait their turn.
    last_job_done_ms: AtomicU64,
    /// Rate-limits the log lines for disk waits that ran out.
    stall_log: StallLog,
    /// Rate-limits the log lines for background jobs that
    /// failed: a disk that refuses them fails each one again
    /// at every retry.
    job_log: StallLog,
    /// How far the directory check and the boot scan have come
    /// (issue #708).
    boot: BootScan,
    /// Deletes of entry files that failed (issue #709): an
    /// evicted entry's, a doomed write's, a corrupt one's, and
    /// the boot scan's (issue #708). Each also counts in
    /// `errors`.
    unlinks_failed: AtomicU64,
    /// Rate-limits the inline ones' log lines: a disk that
    /// refuses deletes refuses one per eviction. The boot scan
    /// logs its own in `boot`.
    unlink_log: StallLog,
    /// Test-only: run at every entry read (after the bytes are
    /// in, as a read stuck in the kernel has its file open),
    /// write and delete, the boot scan's and the directory
    /// probe's included, with the file's path, so a test can
    /// hold or fail the operation the way a stuck or failing
    /// disk would. The store refuses FIFOs (issue #545), so a
    /// test cannot hold a read on one.
    #[cfg(test)]
    hold: Mutex<Option<TestHold>>,
}

/// How long the boot waits for the directory check and the boot
/// scan, from the store's start, before the server goes on
/// without them (issue #708). A scan that ends in time leaves
/// every entry indexed before the server listens, as boot
/// always did. One that does not goes on in the background.
pub const BOOT_SCAN_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// The name of the thread the boot scan runs on.
pub(crate) const SCAN_THREAD: &str = "crabcrop-disk-scan";

/// Where the boot scan is (issue #708).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanPhase {
    /// Creating, probing and opening the directory. Nothing is
    /// read or written yet.
    Checking,
    /// Reading the entry files. Writes land meanwhile.
    Scanning,
    /// What the scan found is indexed.
    Done,
    /// A shutdown stopped the scan before it indexed anything.
    Stopped,
    /// The directory check failed, after the boot went on
    /// without it: the tier stays empty and writes nothing.
    Failed(String),
}

/// How far the boot scan has come (issue #708).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub phase: ScanPhase,
    /// How long the scan has run, or ran.
    pub elapsed: std::time::Duration,
    /// The entry files it has looked at.
    pub files: u64,
    /// The entries it indexed, once done.
    pub entries: usize,
    /// What those entries are charged against the budget.
    pub bytes: usize,
    /// The entry files it deleted: ones that do not verify,
    /// cannot be entries, or no longer fit the budget, and
    /// stale copies of keys written since the boot.
    pub deleted: u64,
    /// The entry files it could not delete. Each also counts
    /// in the tier's errors and in
    /// [`DiskStore::unlinks_failed`].
    pub undeleted: u64,
}

impl ScanPhase {
    /// Whether the scan is over, however it ended.
    pub fn ended(&self) -> bool {
        matches!(
            self,
            ScanPhase::Done | ScanPhase::Stopped | ScanPhase::Failed(_)
        )
    }
}

impl ScanReport {
    /// Whether the scan is over, however it ended.
    pub fn ended(&self) -> bool {
        self.phase.ended()
    }

    /// The entry files it looked at and neither indexed nor
    /// tried to delete: files that are not regular, files gone
    /// before it read them, files a newer copy or a queued
    /// deletion took care of, and files it cannot trust that it
    /// leaves for the next boot because a write of their key
    /// may have replaced them meanwhile.
    pub fn skipped(&self) -> u64 {
        self.files
            .saturating_sub(self.entries as u64)
            .saturating_sub(self.deleted)
            .saturating_sub(self.undeleted)
    }
}

/// The boot scan's progress, as the boot's wait and the log
/// lines read it (issue #708).
#[derive(Debug)]
struct BootScan {
    started: std::time::Instant,
    /// The entry files looked at so far.
    files: AtomicU64,
    /// The entry files deleted so far, and those that could not
    /// be (see [`DiskStore::scan_unlink`]), with the
    /// rate-limited logs of each.
    deleted: AtomicU64,
    undeleted: AtomicU64,
    deleted_log: StallLog,
    undeleted_log: StallLog,
    /// The phase, with what an ended scan indexed and how long
    /// it took.
    state: Mutex<(ScanPhase, usize, usize, Option<std::time::Duration>)>,
    changed: Condvar,
    /// Set by a shutdown: the scan stops at the next file.
    stop: AtomicBool,
}

/// A test hook for [`DiskStore`]'s file operations.
#[cfg(test)]
pub(crate) type TestHold = Arc<dyn Fn(OpKind, &Path) -> std::io::Result<()> + Send + Sync>;

/// The file operations a test can hold (issue #572).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpKind {
    /// An entry read: a lookup, a drain rewrite or the boot
    /// scan's.
    Read,
    /// An entry write and its rename into place, or the boot's
    /// directory probe.
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
    let (file, meta) = open_regular(path).ok()?;
    read_open_capped(file, meta.len(), cap)
}

/// [`read_capped`]'s read of the open `file`, which was `len`
/// bytes long when it was opened (issue #644). A file already
/// past `cap` is refused unread. Otherwise the buffer is sized
/// for the file once: grown from empty, it took a dozen reads
/// and as many reallocations for a 20 KB entry. The read is
/// still capped, so a file that grows past `cap` while it is
/// read is refused as before, and a buffer that cannot be had
/// is a refusal, as it was an error in the read.
fn read_open_capped(mut file: std::fs::File, len: u64, cap: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    if len > cap as u64 {
        return None;
    }
    let mut data = Vec::new();
    data.try_reserve_exact(len as usize).ok()?;
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
///
/// `file://` sources open their file with [`open_regular_at`]
/// (issue #708), relative to the directory that holds it,
/// which the loader reaches from the file root a component at
/// a time (see `load_file` in `loader.rs`).
///
/// Returns the file with the metadata that check read. An
/// error is the open's own, `NotFound` for a path with nothing
/// at it, or one of kind `InvalidInput` for a path that is not
/// a regular file.
#[cfg(unix)]
pub(crate) fn open_regular(path: &Path) -> std::io::Result<(std::fs::File, std::fs::Metadata)> {
    open_regular_at(rustix::fs::CWD, path)
}

/// [`open_regular`] for `path` relative to the open directory
/// `dir`, with the same flags and the same check: the open
/// does not wait for a FIFO's writer, does not follow a
/// symlink, and what it opened must be a regular file.
#[cfg(unix)]
pub(crate) fn open_regular_at(
    dir: impl std::os::fd::AsFd,
    path: impl rustix::path::Arg,
) -> std::io::Result<(std::fs::File, std::fs::Metadata)> {
    use rustix::fs::{Mode, OFlags};
    let fd = rustix::fs::openat(
        dir,
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    let file = std::fs::File::from(fd);
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(not_regular());
    }
    let flags = rustix::fs::fcntl_getfl(&file)?;
    rustix::fs::fcntl_setfl(&file, flags - OFlags::NONBLOCK)?;
    Ok((file, meta))
}

/// Elsewhere there are no FIFOs whose open waits; the type
/// check still keeps reads to regular files.
#[cfg(not(unix))]
pub(crate) fn open_regular(path: &Path) -> std::io::Result<(std::fs::File, std::fs::Metadata)> {
    let file = std::fs::File::open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(not_regular());
    }
    Ok((file, meta))
}

/// What [`open_regular`] answers for a path that is not a
/// regular file.
fn not_regular() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a regular file")
}

/// What the boot scan read at an entry path (issue #708).
enum ScanRead {
    /// An entry that verifies under the store's key.
    Entry {
        key: String,
        groups: Vec<String>,
        written: Option<WriteStamp>,
        file_bytes: usize,
        mtime: std::time::SystemTime,
    },
    /// Nothing is there any more: a deletion got there first.
    Gone,
    /// Something this store cannot trust, and why: deleted on
    /// sight.
    Bad(&'static str),
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

/// The scratch file [`probe_writable`] writes.
const PROBE_NAME: &str = ".crabcrop-write-probe";

/// One boot-time writability probe (issue #367): write a
/// scratch file into `dir` and remove it, so a directory
/// the process cannot write to (a read-only mount, an
/// unprivileged container user with no pre-created
/// directory) fails at store open with a clear message
/// instead of being discovered per request, silently, on
/// every put. The scratch name never matches the entry
/// scan's stem pattern and is removed on success anyway.
fn probe_writable(dir: &Path) -> Result<(), String> {
    let probe = dir.join(PROBE_NAME);
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
    /// indexing whatever entries a previous run left behind
    /// before it returns. `entry_key` is the cache entry MAC
    /// key (see [`crate::url::entry_key`]): files that fail
    /// verification under it are deleted on sight at scan. The
    /// server starts its store with [`DiskStore::start`]
    /// instead, which does this on a thread of its own.
    pub fn open(dir: &Path, max_bytes: usize, entry_key: Vec<u8>) -> Result<Self, String> {
        let store = Self::unscanned(dir, max_bytes, entry_key);
        let read = store.check_dir()?;
        store.scan(read);
        Ok(store)
    }

    /// Start the store rooted at `dir` (issue #708): it is
    /// returned at once, and a thread of its own creates,
    /// probes and opens the directory, then scans it as
    /// [`DiskStore::open`] does. Until the directory has
    /// answered nothing is written, and until the scan ends an
    /// entry it has not indexed is a miss. The boot waits for
    /// both with [`DiskStore::wait_scan`] up to
    /// [`DiskStore::boot_deadline`], so a slow or hung disk
    /// holds it for [`BOOT_SCAN_WAIT`] at most.
    pub fn start(dir: &Path, max_bytes: usize, entry_key: Vec<u8>) -> Arc<Self> {
        let store = Arc::new(Self::unscanned(dir, max_bytes, entry_key));
        store.begin_scan();
        store
    }

    /// A store over `dir` that has not touched it: it writes
    /// nothing until a scan, inline or begun with
    /// [`DiskStore::begin_scan`], has checked the directory.
    pub(crate) fn unscanned(dir: &Path, max_bytes: usize, entry_key: Vec<u8>) -> Self {
        Self {
            dir: dir.to_path_buf(),
            max_bytes,
            entry_key: zeroize::Zeroizing::new(entry_key),
            inner: Mutex::new(Inner::new()),
            errors: std::sync::atomic::AtomicU64::new(0),
            permits: Arc::new(tokio::sync::Semaphore::new(DISK_OP_PERMITS)),
            drain_wake: tokio::sync::Notify::new(),
            progress: tokio::sync::watch::Sender::new(0),
            last_job_done_ms: AtomicU64::new(0),
            stall_log: StallLog::new("disk waits ran out"),
            job_log: StallLog::new("background disk jobs failed"),
            boot: BootScan {
                started: std::time::Instant::now(),
                files: AtomicU64::new(0),
                deleted: AtomicU64::new(0),
                undeleted: AtomicU64::new(0),
                deleted_log: StallLog::new("entry files the boot scan deleted"),
                undeleted_log: StallLog::new("entry files the boot scan could not delete"),
                state: Mutex::new((ScanPhase::Checking, 0, 0, None)),
                changed: Condvar::new(),
                stop: AtomicBool::new(false),
            },
            unlinks_failed: AtomicU64::new(0),
            unlink_log: StallLog::new("entry file deletes failed"),
            #[cfg(test)]
            hold: Mutex::new(None),
        }
    }

    /// Check the directory and scan it on a thread of its own
    /// (see [`DiskStore::start`]). The thread enters the
    /// runtime this is called on, if any, so the deletions the
    /// scan queues start at once.
    pub(crate) fn begin_scan(self: &Arc<Self>) {
        let store = self.clone();
        let runtime = tokio::runtime::Handle::try_current().ok();
        let spawned = std::thread::Builder::new()
            .name(SCAN_THREAD.to_string())
            .spawn(move || {
                let _entered = runtime.as_ref().map(tokio::runtime::Handle::enter);
                match store.check_dir() {
                    Ok(read) => {
                        store.scan(read);
                        store.kick();
                    }
                    Err(e) => store.end_scan(ScanPhase::Failed(e)),
                }
            });
        if let Err(e) = spawned {
            self.end_scan(ScanPhase::Failed(format!(
                "cannot start the result storage scan: {e}"
            )));
        }
    }

    /// Create, probe and open the directory, with the messages
    /// a boot that fails on them gives. Once it answers, writes
    /// may land.
    fn check_dir(&self) -> Result<std::fs::ReadDir, String> {
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| format!("cannot create the result storage directory: {e}"))?;
        self.held(OpKind::Write, &self.dir.join(PROBE_NAME))
            .map_err(|e| {
                format!(
                    "the result storage directory {} is not writable: {e}",
                    self.dir.display()
                )
            })?;
        probe_writable(&self.dir)?;
        let read = std::fs::read_dir(&self.dir)
            .map_err(|e| format!("cannot read the result storage directory: {e}"))?;
        if let Some(scan) = self.inner.lock().unwrap().scan.as_mut() {
            scan.checked = true;
        }
        self.set_phase(ScanPhase::Scanning);
        Ok(read)
    }

    /// When the boot stops waiting for the directory check and
    /// the scan (issue #708).
    pub fn boot_deadline(&self) -> std::time::Instant {
        self.boot.started + BOOT_SCAN_WAIT
    }

    /// How far the boot scan has come.
    pub fn scan_report(&self) -> ScanReport {
        let state = self.boot.state.lock().unwrap();
        self.report(&state)
    }

    fn report(&self, state: &(ScanPhase, usize, usize, Option<std::time::Duration>)) -> ScanReport {
        let (phase, entries, bytes, took) = state;
        ScanReport {
            phase: phase.clone(),
            elapsed: took.unwrap_or_else(|| self.boot.started.elapsed()),
            files: self.boot.files.load(Ordering::Relaxed),
            entries: *entries,
            bytes: *bytes,
            deleted: self.boot.deleted.load(Ordering::Relaxed),
            undeleted: self.boot.undeleted.load(Ordering::Relaxed),
        }
    }

    /// Wait, blocking this thread, until the boot scan's phase
    /// meets `until` or `deadline` passes, and report how far it
    /// came (issue #708).
    pub fn wait_scan(
        &self,
        deadline: std::time::Instant,
        until: impl Fn(&ScanPhase) -> bool,
    ) -> ScanReport {
        let mut state = self.boot.state.lock().unwrap();
        loop {
            let now = std::time::Instant::now();
            if until(&state.0) || now >= deadline {
                return self.report(&state);
            }
            state = self
                .boot
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap()
                .0;
        }
    }

    /// Stop the boot scan if it still runs (issue #708), for a
    /// shutdown: it indexes nothing more and deletes nothing
    /// more, and stops at the next file it reaches. A scan
    /// stuck in the kernel stops once the disk answers. Returns
    /// how many purges sent while it ran it holds, which it then
    /// never applies, so their bucket tombstones stay in force;
    /// `None` when no scan runs.
    pub fn stop_scan(&self) -> Option<usize> {
        self.boot.stop.store(true, Ordering::Relaxed);
        let inner = self.inner.lock().unwrap();
        inner.scan.as_ref().map(|scan| scan.purges.sent)
    }

    fn set_phase(&self, phase: ScanPhase) {
        self.boot.state.lock().unwrap().0 = phase;
        self.boot.changed.notify_all();
    }

    /// End the scan without indexing anything: stopped, or the
    /// directory check failed. Its state goes, so writes are no
    /// longer recorded, and after a failed check they are
    /// refused for good (see [`refuse_write`]). The purges it
    /// held are never applied (see [`ScanPurge`]).
    fn end_scan(&self, phase: ScanPhase) {
        if matches!(phase, ScanPhase::Failed(_)) {
            // Counted like a failed write, so a tier that cannot
            // write shows on /metrics (issue #367). The boot or
            // its watch logs the error.
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(scan) = inner.scan.take()
                && !scan.checked
            {
                // The directory never answered: nothing may be
                // written to it.
                inner.unusable = true;
            }
        }
        let mut state = self.boot.state.lock().unwrap();
        *state = (phase, 0, 0, Some(self.boot.started.elapsed()));
        drop(state);
        self.boot.changed.notify_all();
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
        if let Some(line) = self.job_log.note(monotonic_ms(), &format!("{what}: {e}")) {
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
        if let Some(line) = self.stall_log.note(monotonic_ms(), stall.what()) {
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
    /// deleted until the rest fit, so the budget holds once the
    /// scan ends.
    ///
    /// Requests may be served while it runs (issue #708), so it
    /// reads every file first, leaving the index alone, and
    /// then indexes what it found in one step under the index
    /// lock, judging each file by what was done meanwhile (see
    /// [`ScanState`]). A file it deletes by key goes under the
    /// key's claim, like every other deletion, and one it
    /// deletes by name only once no write can rename into it,
    /// so neither removes a file a write put in place.
    fn scan(&self, read: std::fs::ReadDir) {
        let mut found: Vec<Scanned> = Vec::new();
        let mut overcharged: Vec<(String, String, Vec<String>)> = Vec::new();
        // A zero budget disables the store (reads and writes
        // already refuse); the scan then imposes no size cap of
        // its own.
        let cap = if self.max_bytes == 0 {
            usize::MAX
        } else {
            self.max_bytes
        };
        for entry in read.flatten() {
            if self.boot.stop.load(Ordering::Relaxed) {
                return self.end_scan(ScanPhase::Stopped);
            }
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
            self.boot.files.fetch_add(1, Ordering::Relaxed);
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
                self.unlink_unknown(&stem, &path, "it is larger than the whole budget");
                continue;
            }
            match self.read_scanned(&path, cap) {
                ScanRead::Gone => {}
                ScanRead::Bad(why) => self.unlink_unknown(&stem, &path, why),
                // `put` writes each entry under its own key's
                // stem, so a copy under another name was put
                // there by someone else.
                ScanRead::Entry { key, .. } if key_stem(&key) != stem => {
                    self.unlink_unknown(&stem, &path, "it holds another key's entry");
                }
                ScanRead::Entry {
                    key,
                    groups,
                    written,
                    file_bytes,
                    mtime,
                } => {
                    let charge = entry_charge(&key, file_bytes, &groups);
                    // `put` never stores an entry charged more
                    // than the whole budget. One here was
                    // stored under a larger budget or by a
                    // build that charged the file alone, and
                    // keeping it would leave no room for any
                    // other entry.
                    if self.max_bytes > 0 && charge > self.max_bytes {
                        overcharged.push((key, stem, groups));
                        continue;
                    }
                    found.push(Scanned {
                        mtime,
                        key,
                        stem,
                        file_bytes,
                        charge,
                        groups,
                        written,
                    });
                }
            }
        }
        found.sort_by_key(|scanned| scanned.mtime);
        let Some((unlink, entries, bytes)) = self.index_scanned(found, overcharged) else {
            return self.end_scan(ScanPhase::Stopped);
        };
        for (key, stem, why) in unlink {
            let _claim = FileClaim::inline(self, &key);
            self.scan_unlink(&self.file_for(&stem), why);
        }
        let mut state = self.boot.state.lock().unwrap();
        *state = (
            ScanPhase::Done,
            entries,
            bytes,
            Some(self.boot.started.elapsed()),
        );
        drop(state);
        self.boot.changed.notify_all();
    }

    /// The scan's one step under the index lock (issue #708):
    /// index what it found, oldest first and below every entry
    /// used since the store started, unless what was done while
    /// it read says otherwise (see [`judge_scanned`]), and delete
    /// the oldest until the budget holds. Returns the files to
    /// delete under the claims it took, each with why, and how
    /// many entries it indexed, charged how much; `None` when a
    /// shutdown stopped the scan first. That is checked under
    /// the lock, so the purges [`DiskStore::stop_scan`] reports
    /// as never applied are not applied.
    #[allow(clippy::type_complexity)]
    fn index_scanned(
        &self,
        found: Vec<Scanned>,
        overcharged: Vec<(String, String, Vec<String>)>,
    ) -> Option<(Vec<(String, String, &'static str)>, usize, usize)> {
        let mut unlink = Vec::new();
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        if self.boot.stop.load(Ordering::Relaxed) {
            return None;
        }
        // From here on writes are not recorded: what they
        // replace is indexed.
        let scan = inner.scan.take().unwrap_or_default();
        let mut held: Vec<Vec<u64>> = vec![Vec::new(); scan.purges.holds.len()];
        for (key, stem, groups) in overcharged {
            if !matches!(
                judge_scanned(inner, &scan, &mut held, &key, &stem, &groups),
                Judged::Leave
            ) && let Some(stem) = claim_or_queue(inner, &key, &stem, &groups)
            {
                unlink.push((key, stem, "it is charged more than the whole budget"));
            }
        }
        let mut kept = Vec::with_capacity(found.len());
        for scanned in found {
            let judged = judge_scanned(
                inner,
                &scan,
                &mut held,
                &scanned.key,
                &scanned.stem,
                &scanned.groups,
            );
            match judged {
                Judged::Keep => kept.push(scanned),
                Judged::Delete => {
                    if let Some(stem) =
                        claim_or_queue(inner, &scanned.key, &scanned.stem, &scanned.groups)
                    {
                        unlink.push((
                            scanned.key,
                            stem,
                            "a copy of its key may have been written since the boot",
                        ));
                    }
                }
                Judged::Leave => {}
            }
        }
        // A directory filled under a larger budget, or by a
        // build that charged each entry its file alone, can
        // hold more than the budget. The oldest entries are
        // the ones `put` would evict first: delete them until
        // the rest fit beside what was written since the store
        // started, in one pass over the sorted list.
        let mut total: usize = kept.iter().map(|scanned| scanned.charge).sum();
        let mut over = 0;
        if self.max_bytes > 0 {
            while inner.total_bytes + total > self.max_bytes && over < kept.len() {
                total -= kept[over].charge;
                over += 1;
            }
        }
        for scanned in kept.drain(..over) {
            if let Some(stem) = claim_or_queue(inner, &scanned.key, &scanned.stem, &scanned.groups)
            {
                unlink.push((scanned.key, stem, "it is among the oldest past the budget"));
            }
        }
        let entries = kept.len();
        for (n, scanned) in kept.into_iter().enumerate() {
            let stamp = n as u64 + 1;
            inner.total_bytes += scanned.charge;
            inner.next_id += 1;
            let id = inner.next_id;
            inner.order.insert(stamp, scanned.key.clone());
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
                    stamp,
                },
            );
        }
        for (hold, seqs) in scan.purges.holds.iter().zip(held) {
            let _ = hold.applied.set(seqs);
        }
        Some((unlink, entries, total))
    }

    /// Read the file at an entry path for the boot scan, with
    /// the read hook a lookup runs. The read is capped like any
    /// other entry read: the header parse validates against the
    /// whole file, so the file has to be in memory, but never
    /// more than `cap`. The MAC is verified: a file planted in
    /// the directory is dropped at scan, not indexed. A file
    /// that is gone by the time it is opened is not deleted
    /// (issue #708): a write may have renamed one into place
    /// since.
    fn read_scanned(&self, path: &Path, cap: usize) -> ScanRead {
        let (file, meta) = match open_regular(path) {
            Ok(opened) => opened,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return ScanRead::Gone,
            Err(_) => return ScanRead::Bad("it cannot be opened as a regular file"),
        };
        let data = read_open_capped(file, meta.len(), cap);
        if self.held(OpKind::Read, path).is_err() {
            return ScanRead::Bad("it cannot be read");
        }
        let Some(data) = data else {
            return ScanRead::Bad("it cannot be read whole within the budget");
        };
        let Some(header) = parse_header_bytes(&data, data.len()) else {
            return ScanRead::Bad("it is not an entry this build reads");
        };
        if !verify_entry_mac(&self.entry_key, &data[header.covered_start..], &header.mac) {
            return ScanRead::Bad("it does not verify under this server's secret");
        }
        ScanRead::Entry {
            key: header.key,
            groups: header.groups,
            written: header.written,
            file_bytes: data.len(),
            mtime: meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        }
    }

    /// Delete the file at `path`, an entry path whose key the
    /// boot scan does not know (issue #708): one that does not
    /// verify, or that cannot be an entry, for the reason
    /// `why`. While it goes, a write of a key with its stem
    /// writes nothing, so the deletion cannot remove a file the
    /// write renamed into place. A file at the stem of a key a
    /// write was begun for may be that write's, so it stays: a
    /// landed write replaced it, and otherwise the next boot
    /// deletes it.
    fn unlink_unknown(&self, stem: &str, path: &Path, why: &str) {
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(scan) = inner.scan.as_mut() {
                if scan.written.contains(stem) {
                    return;
                }
                scan.unlinking.insert(stem.to_string());
            }
        }
        self.scan_unlink(path, why);
        if let Some(scan) = self.inner.lock().unwrap().scan.as_mut() {
            scan.unlinking.remove(stem);
        }
    }

    /// Delete the entry file at `path` for the boot scan, for
    /// the reason `why`, under the claim or the stem the caller
    /// holds. The scan deleted silently (issue #708): every file
    /// it deletes now counts in its report, and every one it
    /// cannot delete counts in the tier's errors as well, and
    /// with the inline deletes that fail (issue #709) in
    /// `crabcrop_cache_disk_unlinks_failed_total`. It stays on
    /// disk, out of the index, until the next boot.
    /// Both are logged, each kind the first at once and then at
    /// most one line every 10 seconds with a count of the rest.
    /// A file already gone is neither.
    fn scan_unlink(&self, path: &Path, why: &str) {
        let line = match self
            .held(OpKind::Unlink, path)
            .and_then(|()| std::fs::remove_file(path))
        {
            Ok(()) => {
                self.boot.deleted.fetch_add(1, Ordering::Relaxed);
                self.boot.deleted_log.note(
                    monotonic_ms(),
                    &format!("the boot scan deleted {}: {why}", path.display()),
                )
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                self.boot.undeleted.fetch_add(1, Ordering::Relaxed);
                self.unlinks_failed.fetch_add(1, Ordering::Relaxed);
                self.errors.fetch_add(1, Ordering::Relaxed);
                self.boot.undeleted_log.note(
                    monotonic_ms(),
                    &format!(
                        "the boot scan cannot delete {} ({why}), which stays on disk until the next boot: {e}",
                        path.display()
                    ),
                )
            }
        };
        if let Some(line) = line {
            eprintln!("{line}");
        }
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
                let claimed = evict_entry(&mut inner, key).and_then(|record| {
                    claim_or_queue(&mut inner, key, &record.stem, &record.groups)
                });
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

    /// Record that the bucket copy of `key` lacks groups copy
    /// `id`'s record holds (issue #709): a write-back that was
    /// to carry them was skipped, so the next disk hit sends
    /// them again. No-op when the record is gone or replaced.
    pub fn owe_shared(&self, key: &str, id: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(record) = inner.entries.get_mut(key)
            && record.id == id
        {
            record.owes_shared = true;
        }
    }

    /// Remove `key`'s record if it is still copy `id`, and
    /// queue its file's deletion, retried until it is done. No
    /// file I/O; the caller kicks the drain. A purge that
    /// covers the copy holds its tombstones for the deletion
    /// (see [`DiskStore::purge_waiting`]). Returns whether it
    /// was removed.
    pub fn forget_if(&self, key: &str, id: u64) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if !inner.entries.get(key).is_some_and(|record| record.id == id) {
            return false;
        }
        if let Some(record) = evict_entry(&mut inner, key) {
            let inner = &mut *inner;
            inner
                .deletions
                .push_unheld(key, &record.stem, &record.groups, &mut inner.next_seq);
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
        if let Some(refused) = refuse_write(inner, key) {
            return refused;
        }
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
            match inner.claims.remove(key) {
                Some(Claim::Write {
                    groups,
                    doomed: false,
                }) => {
                    // The rename replaced the file a queued
                    // deletion was for.
                    if inner.deletions.remove(key) {
                        inner.wake_deletion_waits();
                        self.progress.send_modify(|n| *n += 1);
                    }
                    let groups = with_groups(&entry.groups, &groups);
                    if groups.len() > entry.groups.len() {
                        inner.rewrites.push(key, &stem, &mut inner.next_seq);
                    }
                    let charge = entry_charge(key, data.len(), &groups);
                    // The least recently used entry is the first
                    // in the order (issue #708).
                    while inner.total_bytes + charge > self.max_bytes {
                        let Some((_, oldest)) = inner.order.pop_first() else {
                            break;
                        };
                        if let Some(record) = evict_entry(inner, &oldest)
                            && let Some(stem) =
                                claim_or_queue(inner, &oldest, &record.stem, &record.groups)
                        {
                            evicted.push((oldest, stem));
                        }
                    }
                    inner.next_id += 1;
                    let id = inner.next_id;
                    inner.total_bytes += charge;
                    inner.clock += 1;
                    let stamp = inner.clock;
                    inner.order.insert(stamp, key.to_string());
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
                            stamp,
                        },
                    );
                    Some(Landed { id, groups })
                }
                _ => {
                    // Doomed: the file just written goes again,
                    // under the key's claim. The deletion the
                    // purge queued for it stays, so a failed
                    // unlink here is retried, and the purge's
                    // waits hold until the file is gone.
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
            if refuse_write(&mut inner, key).is_some() || inner.claims.contains_key(key) {
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
            record.and_then(|record| claim_or_queue(&mut inner, key, &record.stem, &record.groups))
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
    /// covers is doomed: it lands nothing, and the deletion of
    /// the file it renames into place is queued with the
    /// others, so the purge's waits cover it. Returns how many
    /// records were removed, and the sequence numbers of the
    /// deletions this purge queued, for
    /// [`DiskStore::wait_deletions`]. The caller kicks the
    /// drain.
    pub fn purge(&self, target: &PurgeTarget) -> (usize, Range<u64>) {
        let (removed, seqs, _, _) = self.purge_queued(target);
        (removed, seqs)
    }

    /// [`DiskStore::purge`], with the deletions its bucket
    /// tombstones wait for: those it queued, those queued
    /// before it for keys whose write it doomed, and those of
    /// copies it covers that no purge held yet. A lookup or a
    /// landing that finds a copy under a running purge, while
    /// that purge's tombstone is still being written or after
    /// a restart, removes it and queues its deletion before the
    /// purge gets here; the purge takes that deletion over, so
    /// its tombstones stay in force until the file is gone.
    /// A purge sent while the boot scan runs waits for the scan
    /// to delete the files it covers too (issue #708).
    pub fn purge_waiting(self: &Arc<Self>, target: &PurgeTarget) -> (usize, DeletionWait) {
        let (removed, seqs, earlier, scan) = self.purge_queued(target);
        (
            removed,
            DeletionWait {
                store: self.clone(),
                seqs,
                earlier,
                scan,
            },
        )
    }

    /// [`DiskStore::purge`], plus the sequence numbers of the
    /// deletions queued before it that it holds its tombstones
    /// for: for keys whose write it doomed, and of copies it
    /// covers that no purge held yet, which it takes over. A
    /// key has one deletion queued at a time, so when it has
    /// one already, that one deletes the file the write renames
    /// into place, and the purge queues none of its own. While
    /// the boot scan runs, the purge is also handed to it, for
    /// the files it has not indexed yet (issue #708), and the
    /// last value is what the scan does for it.
    fn purge_queued(
        &self,
        target: &PurgeTarget,
    ) -> (usize, Range<u64>, Vec<u64>, Option<Arc<ScanPurge>>) {
        let mut earlier = Vec::new();
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        let first = inner.next_seq + 1;
        let removed: Vec<(String, Record)> = inner
            .entries
            .extract_if(|key, record| target.covers(key, &record.groups))
            .collect();
        for (key, record) in &removed {
            inner.total_bytes -= record.charge;
            inner.order.remove(&record.stamp);
            if inner
                .rewrites
                .jobs
                .get(key)
                .is_some_and(|job| job.running.is_none())
            {
                inner.rewrites.remove(key);
            }
            inner.deletions.push(key, &record.stem, &mut inner.next_seq);
        }
        let Inner {
            claims,
            deletions,
            next_seq,
            ..
        } = inner;
        for (key, claim) in claims.iter_mut() {
            if let Claim::Write { groups, doomed } = claim
                && target.covers(key, groups)
            {
                *doomed = true;
                match deletions.jobs.get(key) {
                    Some(job) => {
                        earlier.push(job.seq);
                        deletions.unheld.remove(key);
                    }
                    None => deletions.push(key, &key_stem(key), next_seq),
                }
            }
        }
        let KeyQueue { jobs, unheld, .. } = deletions;
        for (key, _) in unheld.extract_if(|key, groups| target.covers(key, groups)) {
            if let Some(job) = jobs.get(&key) {
                earlier.push(job.seq);
            }
        }
        let scan = inner.scan.as_mut().map(|scan| scan.purges.add(target));
        (removed.len(), first..inner.next_seq + 1, earlier, scan)
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
    /// to [`DRAIN_JOBS`] at once, and wake one that waits: a
    /// task for each job that can run and has no task free for
    /// it. Jobs waiting out a backoff need one task that runs
    /// no job to wait for them, since a task running one can be
    /// stuck in the kernel: one starts when every task runs a
    /// job. Without a runtime the work stays queued.
    pub fn kick(self: &Arc<Self>) {
        let (spawn, ready) = {
            let mut inner = self.inner.lock().unwrap();
            let ready = inner.rewrites.ready() + inner.deletions.ready();
            let running = inner.rewrites.running + inner.deletions.running;
            let later = inner.rewrites.next_due().is_some() || inner.deletions.next_due().is_some();
            let spawn = inner.workers < DRAIN_JOBS
                && (ready + running > inner.workers || (inner.workers == running && later));
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
    /// come is waited for by one task that runs no job: a task
    /// running one can be stuck in the kernel.
    async fn drain(self: Arc<Self>) {
        let mut worker = Worker {
            store: &self,
            counted: true,
        };
        loop {
            let Some(permit) = self.permit().await else {
                return;
            };
            let (next, dropped, backoff) = {
                let mut inner = self.inner.lock().unwrap();
                let mut dropped = false;
                let next = next_job(&mut inner, false, &mut dropped);
                if matches!(next, Next::Empty) {
                    inner.workers -= 1;
                    worker.counted = false;
                }
                let backoff =
                    inner.rewrites.next_due().is_some() || inner.deletions.next_due().is_some();
                (next, dropped, backoff)
            };
            if dropped {
                self.progress.send_modify(|n| *n += 1);
            }
            match next {
                Next::Job(job) => {
                    // When this was the task waiting for a retry,
                    // another one waits for it now.
                    if backoff {
                        self.kick();
                    }
                    let claim = FileClaim::job(self.clone(), &job.key, job.kind);
                    let _ = self
                        .blocking_with(permit, move |store| store.run_job(claim, &job))
                        .await;
                }
                Next::Wait(until) => {
                    drop(permit);
                    {
                        let mut inner = self.inner.lock().unwrap();
                        let running = inner.rewrites.running + inner.deletions.running;
                        if inner.workers > running + 1 {
                            // Another task that runs no job waits
                            // instead. Wake it, so it waits for the
                            // earliest retry, this one's included.
                            inner.workers -= 1;
                            worker.counted = false;
                            drop(inner);
                            self.drain_wake.notify_one();
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
                // this job's, under its claim. The job runs
                // until it is deleted, so a delete stuck in the
                // kernel counts as a running job.
                drop(guard);
                self.unlink(path);
                self.inner.lock().unwrap().rewrites.finish(key, false);
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
        }
        self.unlink(path);
        self.inner.lock().unwrap().rewrites.finish(key, false);
    }

    /// Delete an entry file under the claim the caller holds,
    /// best-effort: a file left behind by a failure is one the
    /// next boot indexes again or deletes as corrupt. A failure
    /// is no longer silent (issue #709): it counts in
    /// `crabcrop_cache_disk_unlinks_failed_total` and the
    /// tier's errors, and is logged, the first at once and then
    /// at most one line every 10 seconds. A file already gone
    /// is no failure.
    fn unlink(&self, path: &Path) {
        match self
            .held(OpKind::Unlink, path)
            .and_then(|()| std::fs::remove_file(path))
        {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                self.unlinks_failed.fetch_add(1, Ordering::Relaxed);
                self.errors.fetch_add(1, Ordering::Relaxed);
                let line = format!(
                    "cannot delete a cache entry file; it stays on disk, out of the index, until the next boot finds it: {e}"
                );
                if let Some(line) = self.unlink_log.note_now(&line) {
                    eprintln!("{line}");
                }
            }
        }
    }

    /// Deletes of entry files that failed so far, inline (issue
    /// #709) or the boot scan's (issue #708), surfaced on
    /// /metrics as `crabcrop_cache_disk_unlinks_failed_total`.
    pub fn unlinks_failed(&self) -> u64 {
        self.unlinks_failed.load(Ordering::Relaxed)
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

    /// Wait until the deletions a purge queued (`seqs`) are
    /// done (issue #572), or until none of them has finished for
    /// `idle`, or `cap` has passed in all, and say which (see
    /// [`Waited`]): a disk that stops answering, a slow one, or
    /// one busy with other jobs that come first. Deletions that
    /// failed are not waited for; they are retried and counted.
    /// Only these deletions wake the wait, and each one counts
    /// down what it waits for at a constant cost, so other jobs
    /// ending do not restart the idle time, and neither many
    /// waits nor one purge of many files slows the drain.
    pub async fn wait_deletions(
        self: &Arc<Self>,
        seqs: Range<u64>,
        idle: std::time::Duration,
        cap: std::time::Duration,
    ) -> Waited {
        let count = Arc::new(DeletionCount::default());
        let mut left = {
            let mut guard = self.inner.lock().unwrap();
            let inner = &mut *guard;
            // What ended before this wait is counted out here,
            // so it must not reach the new count as well.
            inner.wake_deletion_waits();
            let left = inner.deletions.count_in(&seqs);
            if left == 0 {
                return Waited::Done;
            }
            count.left.store(left, Ordering::Relaxed);
            inner
                .deletion_waits
                .entry(seqs.start)
                .or_default()
                .push((seqs.end, count.clone()));
            left
        };
        let _registered = DeletionWaitEntry {
            store: self,
            start: seqs.start,
            count: &count,
        };
        self.kick();
        let started = tokio::time::Instant::now();
        let end = started + cap;
        let mut quiet_until = started + idle;
        loop {
            if tokio::time::timeout_at(quiet_until.min(end), count.wake.notified())
                .await
                .is_err()
            {
                return self.stopped_waiting(&seqs, idle);
            }
            let now_left = count.left.load(Ordering::Relaxed);
            if now_left == 0 {
                return Waited::Done;
            }
            if now_left < left {
                left = now_left;
                quiet_until = tokio::time::Instant::now() + idle;
            }
        }
    }

    /// Why a wait on the deletions in `seqs` ran out: the disk
    /// stalled when no background job finished for `idle`, or
    /// when one of these deletions has run for half that long,
    /// which no working disk takes for one deletion. Otherwise
    /// the disk works, and they wait their turn behind other
    /// jobs.
    fn stopped_waiting(&self, seqs: &Range<u64>, idle: std::time::Duration) -> Waited {
        let idle_ms = u64::try_from(idle.as_millis()).unwrap_or(u64::MAX);
        let done_ms = self.last_job_done_ms.load(Ordering::Relaxed);
        if done_ms == 0 || monotonic_ms().saturating_sub(done_ms) >= idle_ms {
            return Waited::Stalled;
        }
        let stuck = std::time::Instant::now()
            .checked_sub(idle / 2)
            .is_some_and(|since| {
                self.inner
                    .lock()
                    .unwrap()
                    .deletions
                    .running_since(seqs, since)
            });
        if stuck {
            Waited::Stalled
        } else {
            Waited::Behind
        }
    }

    /// Whether a deletion in `seqs` is still queued, failed or
    /// not.
    pub fn deletions_pending(&self, seqs: &Range<u64>) -> bool {
        self.inner.lock().unwrap().deletions.any_in(seqs)
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

/// Mark `key`'s entry, if there is one, as used now: its key
/// moves to a new stamp at the back of the order (issue #708).
fn touch(inner: &mut Inner, key: &str) {
    inner.clock += 1;
    let stamp = inner.clock;
    let Inner { entries, order, .. } = inner;
    if let Some(record) = entries.get_mut(key) {
        let before = std::mem::replace(&mut record.stamp, stamp);
        if let Some(key) = order.remove(&before) {
            order.insert(stamp, key);
        }
    }
}

/// Remove `key`'s record from the index, with its charge,
/// its recency and a rewrite queued for it.
fn evict_entry(inner: &mut Inner, key: &str) -> Option<Record> {
    let record = inner.entries.remove(key)?;
    inner.total_bytes -= record.charge;
    inner.order.remove(&record.stamp);
    if inner
        .rewrites
        .jobs
        .get(key)
        .is_some_and(|job| job.running.is_none())
    {
        inner.rewrites.remove(key);
    }
    Some(record)
}

/// Claim the file at `stem` of a copy of `key` in `groups`,
/// just evicted or never indexed, for an inline delete and
/// return its stem, or, when another job holds it, queue the
/// deletion.
fn claim_or_queue(inner: &mut Inner, key: &str, stem: &str, groups: &[String]) -> Option<String> {
    if inner.claims.contains_key(key) {
        inner
            .deletions
            .push_unheld(key, stem, groups, &mut inner.next_seq);
        return None;
    }
    inner.claims.insert(key.to_string(), Claim::File);
    Some(stem.to_string())
}

/// Why a write of `key` may not go on (issue #708):
/// [`WritePlan::Off`] for a directory that never answered or
/// failed its check, and [`WritePlan::Busy`] before the
/// directory has answered or while the scan deletes a file at
/// the key's stem by name. While the scan runs, the write is
/// recorded for it whether it goes on or not: the copy it was
/// for serves from memory either way, under groups a file the
/// scan read for the key may lack.
fn refuse_write(inner: &mut Inner, key: &str) -> Option<WritePlan> {
    if inner.unusable {
        return Some(WritePlan::Off);
    }
    let scan = inner.scan.as_mut()?;
    let stem = key_stem(key);
    let busy = !scan.checked || scan.unlinking.contains(&stem);
    scan.written.insert(&stem);
    busy.then_some(WritePlan::Busy)
}

/// What the boot scan does with a file it read (issue #708).
enum Judged {
    /// Index it.
    Keep,
    /// Delete it under its key's claim.
    Delete,
    /// Nothing: another copy or a queued deletion has it.
    Leave,
}

/// Judge the file the boot scan read at `stem` for a copy of
/// `key` in `groups` by what was done while the scan read
/// (issue #708), under the index lock. An indexed key's
/// record is a write's copy, which replaced the file. A purge
/// sent since the store started that covers the copy gets the
/// file's deletion queued, and its tombstones wait for it,
/// in `held` by purge. A key whose deletion is queued already
/// goes with that. A key a write was begun for, or that a job
/// holds, has a copy that counts more than this one, landed,
/// in flight, or in memory with groups this file lacks: the
/// file goes.
fn judge_scanned(
    inner: &mut Inner,
    scan: &ScanState,
    held: &mut [Vec<u64>],
    key: &str,
    stem: &str,
    groups: &[String],
) -> Judged {
    if inner.entries.contains_key(key) {
        return Judged::Leave;
    }
    let covering = scan.purges.covering(key, groups);
    if !covering.is_empty() {
        let seq = queue_held_deletion(inner, key, stem);
        for at in covering {
            held[at].push(seq);
        }
        return Judged::Leave;
    }
    if inner.deletions.jobs.contains_key(key) {
        return Judged::Leave;
    }
    if scan.written.contains(stem) || inner.claims.contains_key(key) {
        return Judged::Delete;
    }
    Judged::Keep
}

/// Queue the deletion of `key`'s file at `stem` for a purge
/// that holds its tombstones for it, and return its sequence
/// number: that of the deletion queued already, if any, which
/// the purge then holds instead.
fn queue_held_deletion(inner: &mut Inner, key: &str, stem: &str) -> u64 {
    let Inner {
        deletions,
        next_seq,
        ..
    } = inner;
    if let Some(job) = deletions.jobs.get(key) {
        let seq = job.seq;
        deletions.unheld.remove(key);
        return seq;
    }
    deletions.push(key, stem, next_seq);
    *next_seq
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
            // A record that grows is used now.
            inner.clock += 1;
            let stamp = inner.clock;
            let before = std::mem::replace(&mut record.stamp, stamp);
            if let Some(key) = inner.order.remove(&before) {
                inner.order.insert(stamp, key);
            }
        }
        if matches!(bucket, Bucket::Lacks) {
            record.owes_shared = true;
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
            if self.done {
                store
                    .last_job_done_ms
                    .store(monotonic_ms().max(1), Ordering::Relaxed);
            }
            store.progress.send_modify(|n| *n += 1);
        }
    }
}

/// Removes a wait from [`Inner::deletion_waits`] when it ends.
struct DeletionWaitEntry<'s> {
    store: &'s DiskStore,
    start: u64,
    count: &'s Arc<DeletionCount>,
}

impl Drop for DeletionWaitEntry<'_> {
    fn drop(&mut self) {
        let mut inner = self
            .store
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(waits) = inner.deletion_waits.get_mut(&self.start) {
            waits.retain(|(_, count)| !Arc::ptr_eq(count, self.count));
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
    /// A group write-back was skipped past the waiting cap
    /// (issue #709): the bucket copy lacks groups the record
    /// holds, grown now or not, so the record owes them.
    Lacks,
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

/// How a wait on a purge's disk deletions ended (see
/// [`DiskStore::wait_deletions`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waited {
    /// Every one of them that has not failed is done.
    Done,
    /// The wait ran out while other background jobs kept
    /// finishing: the disk works, and the deletions wait their
    /// turn behind those jobs.
    Behind,
    /// The wait ran out on a stuck disk: no background job
    /// finished for the idle time, or one of the deletions has
    /// run for half that long.
    Stalled,
}

/// Milliseconds since the first call, on a clock that a step of
/// the wall clock does not move.
fn monotonic_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let epoch = *EPOCH.get_or_init(std::time::Instant::now);
    u64::try_from(epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
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
    /// A purge's deletions made no progress in time on a disk
    /// that finished no job meanwhile, or on which one of them
    /// is stuck; they go on in the background.
    PurgeDelete,
}

impl Stall {
    fn what(self) -> &'static str {
        match self {
            Stall::LookupSlot => "a disk lookup found no free disk slot in time",
            Stall::LookupRead => "a disk lookup's read did not answer in time",
            Stall::Write => "a disk write found no free disk slot in time and was skipped",
            Stall::PurgeDelete => {
                "a purge's disk deletions stopped making progress on a stuck disk; they go on in the background"
            }
        }
    }
}

/// A rate-limited log of one kind of tier trouble: disk waits
/// that ran out, background jobs or inline deletes that
/// failed, or (issue #709) bucket requests refused. The first
/// line comes at once, then at most one every
/// [`STALL_LOG_EVERY_MS`], with how many were left out since
/// the last one. Its times are [`monotonic_ms`]: on the wall
/// clock, a step back silenced it until the clock passed the
/// old mark.
#[derive(Debug)]
pub(crate) struct StallLog {
    next_line_ms: AtomicU64,
    since: AtomicU64,
    /// The tier the lines speak for, at their start.
    tier: &'static str,
    /// What the left-out ones are, after their count.
    rest: &'static str,
}

impl StallLog {
    /// A log of the disk tier's, `result storage:` lines.
    pub(crate) fn new(rest: &'static str) -> Self {
        Self::for_tier("result storage", rest)
    }

    /// A log whose lines start with `tier`.
    pub(crate) fn for_tier(tier: &'static str, rest: &'static str) -> Self {
        StallLog {
            next_line_ms: AtomicU64::new(0),
            since: AtomicU64::new(0),
            tier,
            rest,
        }
    }

    /// [`StallLog::note`] at the time now.
    pub(crate) fn note_now(&self, what: &str) -> Option<String> {
        self.note(monotonic_ms(), what)
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
            0 => format!("{}: {what}", self.tier),
            more => format!(
                "{}: {what}; {more} more {} since the last line",
                self.tier, self.rest
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
    /// Deletions queued before the purge that it holds its
    /// tombstones for: for keys whose write it doomed, each of
    /// which deletes the file that write renames into place,
    /// and of copies it covers that no purge held.
    pub earlier: Vec<u64>,
    /// For a purge sent while the boot scan ran (issue #708):
    /// what the scan does for it.
    scan: Option<Arc<ScanPurge>>,
}

impl DeletionWait {
    /// Whether any of the deletions, the earlier ones
    /// included, is still queued, failed or not, or, for a
    /// purge sent while the boot scan ran, whether the scan
    /// has yet to delete the files it covers (issue #708).
    pub fn pending(&self) -> bool {
        let inner = self.store.inner.lock().unwrap();
        inner.deletions.any_in(&self.seqs)
            || self
                .earlier
                .iter()
                .any(|seq| inner.deletions.by_seq.contains_key(seq))
            || self
                .scan
                .as_ref()
                .is_some_and(|scan| scan.pending(&inner.deletions))
    }

    /// Wait while the deletions make progress (see
    /// [`DiskStore::wait_deletions`]).
    pub async fn wait(&self, idle: std::time::Duration, cap: std::time::Duration) -> Waited {
        self.store
            .wait_deletions(self.seqs.clone(), idle, cap)
            .await
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
/// [`parse_entry`] and [`DiskStore::read_scanned`], the serving
/// path and the boot scan.
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
/// its own, zeroed when the caller is done with it, like the
/// entry key itself.
fn tombstone_mac_key(entry_key: &[u8]) -> zeroize::Zeroizing<[u8; 32]> {
    zeroize::Zeroizing::new(entry_mac(entry_key, b"crabcrop purge tombstone v1"))
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
    let mac = entry_mac(tombstone_mac_key(entry_key).as_slice(), &body);
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
    if !verify_entry_mac(tombstone_mac_key(entry_key).as_slice(), &data[pos..], &mac) {
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
            let mac = entry_mac(tombstone_mac_key(&key).as_slice(), body.as_bytes());
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
    fn a_failed_inline_delete_is_counted_not_silent() {
        // Issue #709: an evicted entry's file is deleted inline,
        // best-effort, and a delete that failed was silent: the
        // file stayed on disk out of the index, and the next boot
        // indexed it again, with nothing in the log or on
        // /metrics. Each failure now counts, in the tier's errors
        // and on its own series, and is logged at a limited
        // rate. A file that is already gone is no failure.
        let dir = temp_dir("unlink-fails");
        let payload = vec![7u8; 100];
        let file = serialize_entry("a", &image(&payload), &test_entry_key()).len();
        let one = entry_charge("a", file, &[]);
        let store = DiskStore::open(&dir, 2 * one + one / 2, test_entry_key()).unwrap();
        store.put("a", &image(&payload));
        store.put("b", &image(&payload));
        store.set_hold(Some(Arc::new(|kind, _| match kind {
            OpKind::Unlink => Err(std::io::Error::from_raw_os_error(30)),
            _ => Ok(()),
        })));
        let errors = store.errors();
        // `c` evicts `a`, whose file the disk will not delete.
        store.put("c", &image(&payload));
        assert!(store.get("a").is_none());
        assert_eq!(store.unlinks_failed(), 1);
        assert_eq!(store.errors(), errors + 1);
        assert!(dir.join(format!("{}.bin", key_stem("a"))).exists());
        // `d` evicts `b`, whose file is already gone.
        store.set_hold(None);
        std::fs::remove_file(dir.join(format!("{}.bin", key_stem("b")))).unwrap();
        store.put("d", &image(&payload));
        assert!(store.get("b").is_none());
        assert_eq!(store.unlinks_failed(), 1);
        assert_eq!(store.errors(), errors + 1);
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
    fn a_capped_read_takes_a_file_up_to_the_cap_and_no_further() {
        // Issue #644: the read sizes its buffer from the
        // file's length. A file within the cap is read whole,
        // the cap's last byte included, and one past it is
        // refused, as before.
        let dir = temp_dir("read-cap");
        let path = dir.join("entry.bin");
        const CAP: usize = 20_000;
        let bytes: Vec<u8> = (0..CAP + 1).map(|n| (n % 251) as u8).collect();
        for len in [0, 1, 4_096, 8_192, CAP - 1, CAP] {
            std::fs::write(&path, &bytes[..len]).unwrap();
            assert_eq!(
                read_capped(&path, CAP).as_deref(),
                Some(&bytes[..len]),
                "{len} bytes"
            );
        }
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(read_capped(&path, CAP), None, "a byte past the cap");
        assert_eq!(read_capped(&path, CAP + 1), Some(bytes));
        assert_eq!(read_capped(&dir.join("absent.bin"), CAP), None);
    }

    #[test]
    fn a_capped_read_sizes_its_buffer_once() {
        // Issue #644: a buffer that starts empty doubles its
        // way up, a dozen reads and as many reallocations for
        // a 20 KB entry, and ends up to twice the entry's
        // size. Sized from the file's length it is filled as
        // it stands.
        let dir = temp_dir("read-sized");
        let path = dir.join("entry.bin");
        let bytes = vec![7u8; 20_000];
        std::fs::write(&path, &bytes).unwrap();
        let data = read_capped(&path, 1_000_000).unwrap();
        assert_eq!(data, bytes);
        assert!(
            data.capacity() < bytes.len() + bytes.len() / 4,
            "a {}-byte entry was read into a buffer of {}",
            bytes.len(),
            data.capacity()
        );
    }

    #[test]
    fn a_file_that_changes_under_a_capped_read_is_read_within_the_cap_or_refused() {
        // Issue #644: the length is read when the file is
        // opened, and the file can change before it is read.
        // The read stays capped whatever that length said:
        // a file that has grown past the cap since is refused,
        // one that has grown within it or shrunk is read as
        // it now stands (and fails its MAC upstream).
        use std::io::Write;
        let dir = temp_dir("read-grown");
        let path = dir.join("entry.bin");
        const CAP: usize = 10_000;
        let append = |more: usize| {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            file.write_all(&vec![9u8; more]).unwrap();
        };
        // Grown past the cap.
        std::fs::write(&path, vec![1u8; CAP - 100]).unwrap();
        let (file, meta) = open_regular(&path).unwrap();
        let len = meta.len();
        assert_eq!(len, (CAP - 100) as u64);
        append(101);
        assert_eq!(read_open_capped(file, len, CAP), None);
        // Grown, but within the cap.
        std::fs::write(&path, vec![1u8; 4_000]).unwrap();
        let (file, meta) = open_regular(&path).unwrap();
        let len = meta.len();
        append(5_000);
        let data = read_open_capped(file, len, CAP).unwrap();
        assert_eq!(data.len(), 9_000);
        assert_eq!(data, std::fs::read(&path).unwrap());
        // Shrunk.
        std::fs::write(&path, vec![1u8; 4_000]).unwrap();
        let (file, meta) = open_regular(&path).unwrap();
        let len = meta.len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1_000)
            .unwrap();
        assert_eq!(read_open_capped(file, len, CAP), Some(vec![1u8; 1_000]));
        // A length past the cap is refused before any read,
        // whatever the file now holds.
        std::fs::write(&path, vec![1u8; 10]).unwrap();
        let (file, _) = open_regular(&path).unwrap();
        assert_eq!(read_open_capped(file, CAP as u64 + 1, CAP), None);
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

    /// A hook that fails the next unlink, as a disk with a
    /// passing fault does, and lets the rest through.
    fn fail_next_unlink() -> TestHold {
        let failed = std::sync::atomic::AtomicBool::new(false);
        Arc::new(move |op, _| {
            if op == OpKind::Unlink && !failed.swap(true, Ordering::SeqCst) {
                return Err(std::io::Error::from_raw_os_error(5));
            }
            Ok(())
        })
    }

    /// Claim `key` for a write of `bytes` in `group` and land it
    /// on another thread, held in its file write until the
    /// returned sender sends or drops.
    fn held_write(
        store: &Arc<DiskStore>,
        key: &str,
        bytes: &'static [u8],
        group: &str,
    ) -> (
        std::thread::JoinHandle<Option<Landed>>,
        std::sync::mpsc::Sender<()>,
    ) {
        let WritePlan::Claimed(claim) =
            store.begin_write(key, &groups(&[group]), Bucket::Unknown, true)
        else {
            panic!("the key was not claimed");
        };
        let (hook, held, release) = hold_kind(OpKind::Write);
        store.set_hold(Some(hook));
        let writer = {
            let (store, group) = (store.clone(), group.to_string());
            std::thread::spawn(move || store.land(claim, &grouped_image(bytes, &[&group])))
        };
        held.recv().unwrap();
        (writer, release)
    }

    #[test]
    fn a_purge_deletes_the_file_of_a_write_it_dooms_though_the_first_unlink_fails() {
        // Issue #572 second review: a doomed write unlinked its
        // file once when it landed, a failure ignored, and its
        // purge queued nothing, so the purge's waits and its
        // bucket tombstone did not wait for the file, and a
        // restart indexed it in the purged group.
        let dir = temp_dir("doomed-deletion");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        let (writer, release) = held_write(&store, "k", b"v", "g");
        let (removed, seqs) = store.purge(&PurgeTarget::Group("g".into()));
        assert_eq!(removed, 0);
        assert!(store.deletions_pending(&seqs));
        store.set_hold(Some(fail_next_unlink()));
        drop(release);
        assert_eq!(writer.join().unwrap(), None);
        assert!(store.file_for(&key_stem("k")).exists());
        assert!(
            store.deletions_pending(&seqs),
            "the failed unlink is not retried"
        );
        store.set_hold(None);
        store.drain_now();
        assert!(!store.deletions_pending(&seqs));
        assert!(!store.file_for(&key_stem("k")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_doomed_write_keeps_an_earlier_purges_deletion_of_its_key() {
        // The first purge queues the deletion of k's file. A
        // fresh write of k claims it and is doomed by a second
        // purge, so the key's one deletion job stays the first
        // purge's. The doomed landing removed that job, as a
        // landing that stands does, and its own unlink failed:
        // nothing deleted the file, and a restart served it.
        let dir = temp_dir("doomed-earlier");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"v1", &["g"]));
        let (_, first) = store.purge(&PurgeTarget::Group("g".into()));
        let (writer, release) = held_write(&store, "k", b"v2", "g");
        store.purge(&PurgeTarget::Group("g".into()));
        store.set_hold(Some(fail_next_unlink()));
        drop(release);
        assert_eq!(writer.join().unwrap(), None);
        assert!(store.deletions_pending(&first));
        store.set_hold(None);
        store.drain_now();
        assert!(!store.file_for(&key_stem("k")).exists());
        drop(store);
        let reopened = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert!(
            reopened.get("k").is_none(),
            "a restart serves the doomed copy"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_deletion_queued_outside_a_purge_is_held_by_the_first_purge_that_covers_it() {
        // Issue #572 fourth review: a lookup that finds a copy
        // under a purge removes it and queues its deletion before
        // the purge's disk step, or after a restart. The first
        // purge that covers the copy takes that deletion over and
        // holds its tombstones for it; a purge that does not
        // cover it, and a repeat purge, record nothing.
        let dir = temp_dir("unheld-taken-over");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("k", &grouped_image(b"v1", &["g"]));
        let (_, id) = store.get_with_id("k").unwrap();
        assert!(store.forget_if("k", id));
        let (_, other) = store.purge_waiting(&PurgeTarget::Group("h".into()));
        assert!(other.earlier.is_empty() && !other.pending());
        let (_, first) = store.purge_waiting(&PurgeTarget::Group("g".into()));
        assert_eq!(first.earlier.len(), 1);
        assert!(first.pending());
        let (_, repeat) = store.purge_waiting(&PurgeTarget::Group("g".into()));
        assert!(repeat.earlier.is_empty() && !repeat.pending());
        store.drain_now();
        assert!(!store.file_for(&key_stem("k")).exists());
        assert!(!first.pending());
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
                let waited = store
                    .wait_deletions(
                        seqs,
                        std::time::Duration::from_secs(1),
                        std::time::Duration::from_secs(30),
                    )
                    .await;
                (waited == Waited::Done, started.elapsed())
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
            inner.order.insert(inner.clock, key.clone());
            inner.total_bytes += charge;
            let record = Record {
                stem: key_stem(&key),
                file_bytes: 10,
                charge,
                groups,
                written: None,
                id: inner.next_id,
                owes_shared: false,
                stamp: inner.clock,
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
        let waited = store
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
        // Other jobs finish, but this one has hung.
        assert_eq!(waited, Waited::Stalled);
        assert!(took < std::time::Duration::from_secs(3), "{took:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// How long the queued deletions of `purges` purges of
    /// `files` keys each take to drain once a stuck disk
    /// answers, with `waits` waits on each purge's deletions.
    async fn drain_after_a_stall(purges: usize, files: usize, waits: usize) -> std::time::Duration {
        let dir = temp_dir(&format!("drain-waiters-{purges}-{files}-{waits}"));
        let store = Arc::new(DiskStore::open(&dir, usize::MAX / 4, test_entry_key()).unwrap());
        index_without_files(&store, purges * files, |i| format!("g{}", i / files));
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
        let mut tasks = Vec::new();
        for g in 0..purges {
            let (_, seqs) = store.purge(&PurgeTarget::Group(format!("g{g}")));
            for _ in 0..waits {
                let (store, seqs) = (store.clone(), seqs.clone());
                tasks.push(tokio::spawn(async move {
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
        for task in tasks {
            assert_eq!(task.await.unwrap(), Waited::Done);
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
        let alone = drain_after_a_stall(1_000, 4, 0).await;
        let waited = drain_after_a_stall(1_000, 4, 1).await;
        assert!(
            waited < alone * 5 + std::time::Duration::from_secs(1),
            "{alone:?} with no waits, {waited:?} with 1,000"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_purge_waiting_on_its_own_many_deletions_does_not_slow_them() {
        // Issue #572 second review: each of a purge's deletions
        // woke its waits, and each wake counted what was left of
        // the purge's whole range under the index lock, so one
        // purge of n files cost n²/2 steps there. 10,000 took
        // 20 s to drain under the request's wait and the bucket
        // tombstone's, against 0.3 s with no wait.
        let alone = drain_after_a_stall(1, 10_000, 0).await;
        let waited = drain_after_a_stall(1, 10_000, 2).await;
        assert!(
            waited < alone * 3 + std::time::Duration::from_secs(1),
            "{alone:?} with no waits, {waited:?} with two"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_job_queued_while_another_runs_starts_at_once() {
        // A job that can run gets a drain task of its own while
        // another job holds one, up to the limit.
        let dir = temp_dir("parallel-jobs");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("a", &image(b"1"));
        store.put("b", &image(b"2"));
        let (hook, held, release) = hold_kind(OpKind::Write);
        store.set_hold(Some(hook));
        store.learn_groups("a", &groups(&["g"]), Bucket::Unknown);
        store.kick();
        tokio::task::spawn_blocking(move || {
            held.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
        })
        .await
        .unwrap();
        let (_, seqs) = store.purge(&PurgeTarget::Key("b".into()));
        let waited = store
            .wait_deletions(
                seqs,
                std::time::Duration::from_secs(2),
                std::time::Duration::from_secs(5),
            )
            .await;
        store.set_hold(None);
        drop(release);
        assert_eq!(
            waited,
            Waited::Done,
            "the deletion waited for the held rewrite"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_job_runs_again_when_its_backoff_ends_while_another_job_is_stuck() {
        // Issue #572 second review: a drain task that found only
        // a job waiting out its backoff stopped whenever another
        // task ran, even one stuck in the kernel, and kicks start
        // no task for such a job, so its retry waited for the
        // stuck job to return.
        let dir = temp_dir("retry-beside-stuck");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("stuck", &image(b"s"));
        store.put("flaky", &image(b"f"));
        let stuck = store.file_for(&key_stem("stuck"));
        let flaky = store.file_for(&key_stem("flaky"));
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Mutex::new(released);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let flaky_path = flaky.clone();
        store.set_hold(Some(Arc::new(move |kind, path| {
            if kind == OpKind::Unlink && path == stuck {
                let _ = released
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(20));
            }
            if kind == OpKind::Unlink && path == flaky_path && !failed.swap(true, Ordering::SeqCst)
            {
                return Err(std::io::Error::from_raw_os_error(5));
            }
            Ok(())
        })));
        store.purge(&PurgeTarget::Key("stuck".into()));
        store.kick();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.inner.lock().unwrap().deletions.running == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        store.purge(&PurgeTarget::Key("flaky".into()));
        store.kick();
        // Its first run fails, and the retry is due a second later.
        let retried = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while flaky.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        drop(release);
        store.set_hold(None);
        assert!(
            retried.is_ok(),
            "the failed deletion waited for the stuck one: {:?}",
            store.backlog()
        );
        let _ = std::fs::remove_dir_all(&dir);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_job_runs_again_when_the_task_waiting_for_it_takes_a_job_that_gets_stuck() {
        // Issue #572 third review: the one drain task waiting
        // out a failed job's backoff was woken for a new job,
        // took it, and that job got stuck in the kernel. No
        // task was left to wait, and kicks start none for a job
        // in its backoff while a task runs, so the retry waited
        // for the stuck job to return.
        let dir = temp_dir("retry-waiter-stuck");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("flaky", &image(b"f"));
        store.put("stuck", &image(b"s"));
        let flaky = store.file_for(&key_stem("flaky"));
        let stuck = store.file_for(&key_stem("stuck"));
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Mutex::new(released);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let flaky_path = flaky.clone();
        store.set_hold(Some(Arc::new(move |kind, path| {
            if kind == OpKind::Unlink && path == stuck {
                let _ = released
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(20));
            }
            if kind == OpKind::Unlink && path == flaky_path && !failed.swap(true, Ordering::SeqCst)
            {
                return Err(std::io::Error::from_raw_os_error(5));
            }
            Ok(())
        })));
        store.purge(&PurgeTarget::Key("flaky".into()));
        store.kick();
        // Its first run fails, and the retry is due a second
        // later.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.inner.lock().unwrap().deletions.failing == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        store.purge(&PurgeTarget::Key("stuck".into()));
        store.kick();
        let retried = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while flaky.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        drop(release);
        store.set_hold(None);
        assert!(
            retried.is_ok(),
            "the failed deletion waited for the stuck one: {:?}",
            store.backlog()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_job_runs_again_while_a_rewrite_deletes_the_file_of_a_purged_record() {
        // Issue #572 third review: a rewrite whose record a
        // purge removed while it wrote the file ended its job
        // before deleting that file, so a task stuck in that
        // delete counted as free, and the task waiting out a
        // failed job's backoff stopped for it.
        let dir = temp_dir("retry-rewrite-unlink");
        let store = Arc::new(DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap());
        store.put("flaky", &image(b"f"));
        store.put("k", &grouped_image(b"k", &["a"]));
        let flaky = store.file_for(&key_stem("flaky"));
        let k_path = store.file_for(&key_stem("k"));
        let k_stem = key_stem("k");
        let (write_go, write_wait) = std::sync::mpsc::channel::<()>();
        let (unlink_go, unlink_wait) = std::sync::mpsc::channel::<()>();
        let (seen_tx, seen_rx) = std::sync::mpsc::channel::<OpKind>();
        let write_wait = Mutex::new(write_wait);
        let unlink_wait = Mutex::new(unlink_wait);
        let seen_tx = Mutex::new(seen_tx);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let flaky_path = flaky.clone();
        store.set_hold(Some(Arc::new(move |kind, path| {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if kind == OpKind::Write && name.starts_with(&k_stem) {
                let _ = seen_tx.lock().unwrap().send(kind);
                let _ = write_wait
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(20));
            }
            if kind == OpKind::Unlink && path == k_path {
                let _ = seen_tx.lock().unwrap().send(kind);
                let _ = unlink_wait
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(20));
            }
            if kind == OpKind::Unlink && path == flaky_path && !failed.swap(true, Ordering::SeqCst)
            {
                return Err(std::io::Error::from_raw_os_error(5));
            }
            Ok(())
        })));
        // A rewrite of k runs and is held in its write.
        assert!(grew(store.learn_groups(
            "k",
            &groups(&["b"]),
            Bucket::Unknown
        )));
        store.kick();
        let seen = seen_rx.recv_timeout(std::time::Duration::from_secs(5));
        assert_eq!(seen.unwrap(), OpKind::Write);
        // flaky's deletion fails once, and another task waits
        // for its retry.
        store.purge(&PurgeTarget::Key("flaky".into()));
        store.kick();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while store.inner.lock().unwrap().deletions.failing == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        // k is purged while its rewrite writes; the rewrite then
        // finds the record gone and deletes the file, and that
        // delete is stuck.
        store.purge(&PurgeTarget::Key("k".into()));
        store.kick();
        drop(write_go);
        let seen = seen_rx.recv_timeout(std::time::Duration::from_secs(5));
        assert_eq!(seen.unwrap(), OpKind::Unlink);
        store.kick();
        let retried = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while flaky.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        drop(unlink_go);
        store.set_hold(None);
        assert!(
            retried.is_ok(),
            "the failed deletion waited for the stuck rewrite: {:?}",
            store.backlog()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Issue #708: the boot scan on a thread of its own, and the
    // index ordered by last use.

    /// Holds the boot scan's file operations of the given kinds,
    /// on the scan's own thread only, so the test's lookups and
    /// writes go on meanwhile (issue #708). Each operation held
    /// is reported with its path and waits until the gate opens
    /// or drops, or a backstop passes, so a failed test cannot
    /// leave the scan thread waiting.
    struct ScanGate {
        open: Arc<(Mutex<bool>, Condvar)>,
        held: std::sync::mpsc::Receiver<(OpKind, PathBuf)>,
    }

    impl ScanGate {
        fn hold(store: &DiskStore, kinds: &[OpKind]) -> Self {
            let open = Arc::new((Mutex::new(false), Condvar::new()));
            let (held_tx, held) = std::sync::mpsc::channel();
            let held_tx = Mutex::new(held_tx);
            let kinds = kinds.to_vec();
            let gate = open.clone();
            store.set_hold(Some(Arc::new(move |kind, path| {
                if kinds.contains(&kind) && std::thread::current().name() == Some(SCAN_THREAD) {
                    let _ = held_tx.lock().unwrap().send((kind, path.to_path_buf()));
                    let (flag, wake) = &*gate;
                    let opened = flag.lock().unwrap();
                    let _ = wake
                        .wait_timeout_while(opened, std::time::Duration::from_secs(20), |open| {
                            !*open
                        })
                        .unwrap();
                }
                Ok(())
            })));
            ScanGate { open, held }
        }

        /// The next operation the scan is held at.
        fn next(&self) -> (OpKind, PathBuf) {
            self.held
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the boot scan reached no operation the gate holds")
        }

        fn release(&self) {
            let (flag, wake) = &*self.open;
            *flag.lock().unwrap() = true;
            wake.notify_all();
        }
    }

    impl Drop for ScanGate {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// A store over `dir` whose boot scan runs on its own thread
    /// under a gate that holds the given operations.
    fn started_under_gate(
        dir: &Path,
        max_bytes: usize,
        kinds: &[OpKind],
    ) -> (Arc<DiskStore>, ScanGate) {
        let store = Arc::new(DiskStore::unscanned(dir, max_bytes, test_entry_key()));
        let gate = ScanGate::hold(&store, kinds);
        store.begin_scan();
        (store, gate)
    }

    fn scan_ended(store: &DiskStore) -> ScanReport {
        store.wait_scan(
            std::time::Instant::now() + std::time::Duration::from_secs(10),
            ScanPhase::ended,
        )
    }

    /// Store `keys` with `payload` through a store that is then
    /// dropped, as a previous run leaves them, and give their
    /// files modification times in that order, oldest first.
    fn left_by_a_previous_run(dir: &Path, keys: &[String], payload: &[u8]) {
        let store = DiskStore::open(dir, 1_000_000, test_entry_key()).unwrap();
        for (i, key) in keys.iter().enumerate() {
            store.put(key, &image(payload));
            let when = std::time::SystemTime::UNIX_EPOCH
                + std::time::Duration::from_secs(1_000_000 + i as u64);
            std::fs::File::options()
                .write(true)
                .open(store.file_for(&key_stem(key)))
                .unwrap()
                .set_modified(when)
                .unwrap();
        }
    }

    /// The indexed keys from the least recently used, checking
    /// on the way that the order and the records name each other
    /// and that the total is the sum of the charges.
    fn lru_order(store: &DiskStore) -> Vec<String> {
        let inner = store.inner.lock().unwrap();
        assert_eq!(inner.order.len(), inner.entries.len());
        for (stamp, key) in &inner.order {
            assert_eq!(inner.entries[key].stamp, *stamp, "{key}");
        }
        assert_eq!(
            inner.total_bytes,
            inner.entries.values().map(|r| r.charge).sum::<usize>()
        );
        inner.order.values().cloned().collect()
    }

    #[test]
    fn a_held_boot_scan_lets_the_store_miss_and_write_and_then_indexes_below_those_writes() {
        // Issue #708: the scan read and verified every entry
        // before the server listened, so a slow or hung disk
        // kept it from listening. It runs on a thread of its own
        // now. While it is held, the boot's wait ends at its
        // deadline, an entry the scan has not indexed is a miss,
        // and a write lands and serves. Once the scan ends, what
        // it found is indexed, oldest first, as older than that
        // write.
        let dir = temp_dir("scan-held");
        let old: Vec<String> = (0..3).map(|i| format!("old/{i}")).collect();
        left_by_a_previous_run(&dir, &old, b"old");
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Read]);
        gate.next();
        let started = std::time::Instant::now();
        let report = store.wait_scan(
            started + std::time::Duration::from_millis(200),
            ScanPhase::ended,
        );
        assert_eq!(report.phase, ScanPhase::Scanning);
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        for key in &old {
            assert!(store.get(key).is_none(), "{key} is a miss until indexed");
        }
        store.put("new", &image(b"fresh"));
        assert_eq!(store.get("new").unwrap().bytes.as_ref(), b"fresh");
        gate.release();
        let report = scan_ended(&store);
        assert_eq!(report.phase, ScanPhase::Done);
        assert_eq!(report.entries, 3);
        assert!(report.files >= 3, "{report:?}");
        let mut expected = old.clone();
        expected.push("new".to_string());
        assert_eq!(lru_order(&store), expected);
        for key in &old {
            assert_eq!(store.get(key).unwrap().bytes.as_ref(), b"old");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_purge_sent_while_the_boot_scan_runs_deletes_the_files_it_covers_instead_of_indexing_them()
    {
        // Issue #708: an entry purged before the scan reaches it
        // must not come back. The purge finds nothing indexed;
        // the scan, which read the files first, deletes the ones
        // it covers instead of indexing them, and the purge's
        // tombstones wait for those deletions, so a restart
        // before they are done finds the purge still in force.
        let dir = temp_dir("scan-purge");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("a", &grouped_image(b"a", &["g"]));
            store.put("b", &image(b"b"));
            store.put("c", &grouped_image(b"c", &["g"]));
        }
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Read]);
        gate.next();
        let (removed, wait) = store.purge_waiting(&PurgeTarget::Group("g".into()));
        assert_eq!(removed, 0, "nothing is indexed yet");
        assert!(wait.pending(), "the scan has yet to apply the purge");
        let (_, again) = store.purge_waiting(&PurgeTarget::Group("g".into()));
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        assert_eq!(store.keys(), vec!["b".to_string()]);
        assert!(store.get("a").is_none() && store.get("c").is_none());
        assert!(wait.pending(), "the purge waits for the scan's deletions");
        assert!(again.pending(), "so does the same purge sent again");
        store.drain_now();
        assert!(!wait.pending() && !again.pending());
        for key in ["a", "c"] {
            assert!(!store.file_for(&key_stem(key)).exists(), "{key}");
        }
        drop(store);
        let reopened = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(reopened.keys(), vec!["b".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writes_while_the_boot_scan_runs_keep_their_copies_and_the_budget_holds_once_it_ends() {
        // Issue #708: a write that lands while the scan runs
        // replaces the file the scan read for its key, and evicts
        // as a write always does. The scan does not index what
        // it read for that key, indexes the rest below the
        // writes, and deletes the oldest until the budget holds
        // beside them.
        let dir = temp_dir("scan-writes");
        let payload = [7u8; 200];
        let old: Vec<String> = (0..5).map(|i| format!("old/{i}")).collect();
        left_by_a_previous_run(&dir, &old, &payload);
        let file = serialize_entry(&old[0], &image(&payload), &test_entry_key()).len();
        let one = entry_charge(&old[0], file, &[]);
        let budget = 4 * one + one / 2;
        let (store, gate) = started_under_gate(&dir, budget, &[OpKind::Read]);
        let (_, held) = gate.next();
        let rewritten = old
            .iter()
            .find(|key| store.file_for(&key_stem(key)) == held)
            .unwrap()
            .clone();
        // The scan has read this key's file and waits; a fresh
        // copy replaces it, and a new key lands beside it.
        store.put(&rewritten, &image(b"fresh copy"));
        store.put("new", &image(&payload));
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        let order = lru_order(&store);
        let (entries, total) = store.stats();
        assert!(total <= budget, "{entries} entries, {total} over {budget}");
        // The newest of the other files fill what the writes
        // leave, and the older ones are deleted.
        let others: Vec<&String> = old.iter().filter(|key| **key != rewritten).collect();
        let mut expected: Vec<String> = others[2..].iter().map(|key| key.to_string()).collect();
        expected.push(rewritten.clone());
        expected.push("new".to_string());
        assert_eq!(order, expected);
        for key in &others[..2] {
            assert!(!store.file_for(&key_stem(key)).exists(), "{key}");
        }
        assert_eq!(store.get(&rewritten).unwrap().bytes.as_ref(), b"fresh copy");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_entry_evicted_while_the_boot_scan_runs_does_not_come_back_from_the_file_it_read() {
        // Issue #708: writes that land while the scan runs evict
        // as always. A key the scan read a file for, written
        // afresh and then evicted, has no file left, and the scan
        // must not index the copy it read: it would charge the
        // budget for a file that is gone. The files left are
        // small and the writes large, so every file the scan
        // read would fit beside the writes.
        let dir = temp_dir("scan-evict");
        let small = [7u8; 10];
        let large = [7u8; 2_000];
        let old: Vec<String> = (0..3).map(|i| format!("old/{i}")).collect();
        left_by_a_previous_run(&dir, &old, &small);
        let charge = |payload: &[u8]| {
            let file = serialize_entry(&old[0], &image(payload), &test_entry_key()).len();
            entry_charge(&old[0], file, &[])
        };
        let budget = 2 * charge(&large) + 3 * charge(&small) + charge(&small) / 2;
        assert!(3 * charge(&large) > budget);
        let (store, gate) = started_under_gate(&dir, budget, &[OpKind::Read]);
        let (_, held) = gate.next();
        let evicted = old
            .iter()
            .find(|key| store.file_for(&key_stem(key)) == held)
            .unwrap()
            .clone();
        store.put(&evicted, &image(&large));
        store.put("new/1", &image(&large));
        store.put("new/2", &image(&large));
        assert!(
            !store.contains(&evicted),
            "the third write evicted the first"
        );
        assert!(!held.exists());
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        let mut expected: Vec<String> =
            old.iter().filter(|key| **key != evicted).cloned().collect();
        expected.push("new/1".to_string());
        expected.push("new/2".to_string());
        assert_eq!(lru_order(&store), expected);
        assert!(store.get(&evicted).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_the_boot_scan_cannot_trust_is_deleted_only_while_no_write_can_land_over_it() {
        // Issue #708: the scan deletes a file that does not
        // verify by name, not knowing its key, while writes run.
        // A write of that key that lands first renamed a good
        // copy over it, so the scan leaves the file of a key a
        // write was begun for, and while it deletes one, a write
        // of its key writes nothing.
        let bad = |dir: &Path, key: &str| {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(
                dir.join(format!("{}.bin", key_stem(key))),
                b"CRABDISK6\nnot an entry",
            )
            .unwrap();
        };
        // The scan has read the bad file when the write lands.
        let dir = temp_dir("scan-bad-landed");
        bad(&dir, "k");
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Read]);
        gate.next();
        store.put("k", &image(b"fresh"));
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        assert_eq!(store.get("k").unwrap().bytes.as_ref(), b"fresh");
        let _ = std::fs::remove_dir_all(&dir);
        // The scan deletes the bad file when the write begins.
        let dir = temp_dir("scan-bad-deleting");
        bad(&dir, "k");
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Unlink]);
        gate.next();
        assert!(matches!(
            store.begin_write("k", &[], Bucket::Unknown, true),
            WritePlan::Busy
        ));
        store.put("k", &image(b"fresh"));
        assert!(!store.contains("k"), "nothing was written");
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        assert!(!store.file_for(&key_stem("k")).exists());
        store.put("k", &image(b"fresh"));
        assert_eq!(store.get("k").unwrap().bytes.as_ref(), b"fresh");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_shutdown_stops_the_boot_scan_and_leaves_the_purges_it_held_pending() {
        // Issue #708: a shutdown during the scan stops it at its
        // next file. It indexes and deletes nothing more, the
        // files stay for the next boot's scan, and the purges
        // sent while it ran stay pending, so their bucket
        // tombstones are never marked ended.
        let dir = temp_dir("scan-stop");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("a", &grouped_image(b"a", &["g"]));
            store.put("b", &image(b"b"));
        }
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Read]);
        gate.next();
        let (_, wait) = store.purge_waiting(&PurgeTarget::Group("g".into()));
        let (_, again) = store.purge_waiting(&PurgeTarget::Group("g".into()));
        assert_eq!(store.stop_scan(), Some(2));
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Stopped);
        assert!(store.keys().is_empty());
        assert!(wait.pending() && again.pending());
        assert_eq!(store.stop_scan(), None);
        for key in ["a", "b"] {
            assert!(store.file_for(&key_stem(key)).exists(), "{key}");
        }
        store.put("c", &image(b"c"));
        assert_eq!(store.get("c").unwrap().bytes.as_ref(), b"c");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_that_does_not_answer_holds_every_write_until_it_does() {
        // Issue #708: the boot waits for the directory check up
        // to its deadline only. Until the directory answers,
        // nothing is written to it.
        let dir = temp_dir("scan-check");
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Write]);
        let (kind, path) = gate.next();
        assert_eq!(kind, OpKind::Write);
        assert!(path.ends_with(PROBE_NAME), "{}", path.display());
        let report = store.wait_scan(
            std::time::Instant::now() + std::time::Duration::from_millis(200),
            |phase| *phase != ScanPhase::Checking,
        );
        assert_eq!(report.phase, ScanPhase::Checking);
        assert!(matches!(
            store.begin_write("k", &[], Bucket::Unknown, true),
            WritePlan::Busy
        ));
        store.put("k", &image(b"k"));
        assert!(!store.contains("k"));
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        store.put("k", &image(b"k"));
        assert_eq!(store.get("k").unwrap().bytes.as_ref(), b"k");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_check_that_fails_in_the_background_leaves_the_tier_off_and_counted() {
        // Issue #708: a check that fails once the boot went on
        // without it writes nothing, ever, and counts as an
        // error, so /metrics shows the tier failing.
        let dir = temp_dir("scan-check-fails");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("file"), b"").unwrap();
        let store = Arc::new(DiskStore::unscanned(
            &dir.join("file").join("cache"),
            1_000_000,
            test_entry_key(),
        ));
        store.begin_scan();
        let report = scan_ended(&store);
        assert!(
            matches!(&report.phase, ScanPhase::Failed(e) if e.contains("cannot create the result storage directory")),
            "{report:?}"
        );
        assert!(matches!(
            store.begin_write("k", &[], Bucket::Unknown, true),
            WritePlan::Off
        ));
        assert_eq!(store.errors(), 1);
    }

    #[test]
    fn the_boot_scan_deletes_an_entry_stored_under_another_keys_name() {
        // Issue #708: `put` writes each entry under its own key's
        // stem. A verified entry under another name was copied
        // there by someone else; the scan deletes it like a file
        // that does not verify, so every record's file is the
        // one a write of its key replaces.
        let dir = temp_dir("scan-misnamed");
        let copy = {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("k", &image(b"payload"));
            let copy = store.file_for(&key_stem("other"));
            std::fs::copy(store.file_for(&key_stem("k")), &copy).unwrap();
            copy
        };
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(store.keys(), vec!["k".to_string()]);
        assert!(!copy.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_refused_before_the_directory_answers_keeps_the_scan_from_indexing_the_old_file() {
        // Issue #708: until the directory answers, a write is
        // refused, but the copy it was for still serves from
        // memory, under groups the file a previous run left for
        // its key may lack. Indexed, that file would be a copy a
        // purge of those groups does not reach, so the scan
        // treats the key as written: it deletes the file, and
        // indexes the others.
        let dir = temp_dir("scan-check-refused");
        left_by_a_previous_run(&dir, &["k".to_string(), "other".to_string()], b"old");
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Write]);
        let (_, path) = gate.next();
        assert!(path.ends_with(PROBE_NAME), "{}", path.display());
        let groups = ["g".to_string()];
        assert!(matches!(
            store.begin_write("k", &groups, Bucket::Unknown, true),
            WritePlan::Busy
        ));
        assert!(matches!(
            store.learn_groups("k", &groups, Bucket::Unknown),
            Learned::Absent
        ));
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        assert_eq!(store.keys(), vec!["other".to_string()]);
        assert!(!store.file_for(&key_stem("k")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// What the boot scan holds, in bytes, for the keys written
    /// while it runs.
    fn written_bytes(store: &DiskStore) -> usize {
        let inner = store.inner.lock().unwrap();
        let written = &inner.scan.as_ref().unwrap().written;
        written.bits.capacity() * size_of::<u64>()
    }

    #[test]
    fn a_scan_that_outlasts_many_writes_holds_a_fixed_amount_of_memory_for_them() {
        // Issue #708: the scan remembers every key written while
        // it runs, refused writes included. On a disk that hangs
        // it runs as long as the hang, and every key rendered
        // meanwhile is written, so what it holds for them must
        // not grow with their number. It still finds each one.
        let dir = temp_dir("scan-many-writes");
        left_by_a_previous_run(&dir, &["kept".to_string(), "w/77777".to_string()], b"old");
        let (store, gate) = started_under_gate(&dir, 1_000_000, &[OpKind::Write]);
        gate.next();
        let write = |key: &str| {
            assert!(matches!(
                store.begin_write(key, &[], Bucket::Unknown, true),
                WritePlan::Busy
            ));
        };
        write("w/0");
        let first = written_bytes(&store);
        assert!(first <= 2 << 20, "{first} bytes for one key");
        for n in 1..100_000 {
            write(&format!("w/{n}"));
        }
        assert_eq!(written_bytes(&store), first);
        gate.release();
        assert_eq!(scan_ended(&store).phase, ScanPhase::Done);
        assert_eq!(store.keys(), vec!["kept".to_string()]);
        assert!(!store.file_for(&key_stem("w/77777")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_stem_filter_never_misses_a_stem_and_seldom_claims_another() {
        // Issue #708: the filter that keeps the stems written
        // while the scan runs answers yes for every stem it
        // holds, so the scan never indexes a written key's old
        // file. It may answer yes for a stem it does not hold,
        // which only costs that file; at 100,000 stems that
        // happens about once in three million.
        let mut filter = StemFilter::default();
        assert!(!filter.contains(&key_stem("k")));
        assert!(filter.bits.is_empty(), "nothing is held before a stem");
        let held: Vec<String> = (0..100_000)
            .map(|n| key_stem(&format!("held/{n}")))
            .collect();
        for stem in &held {
            filter.insert(stem);
        }
        assert_eq!(filter.bits.len() * 64, StemFilter::BITS);
        assert!(held.iter().all(|stem| filter.contains(stem)));
        let claimed = (0..100_000)
            .filter(|n| filter.contains(&key_stem(&format!("other/{n}"))))
            .count();
        assert!(claimed <= 2, "{claimed} of 100000 others claimed");
        // A stem in capitals, as a file name can spell one, is
        // the same stem.
        assert!(filter.contains(&held[0].to_uppercase()));
    }

    #[test]
    fn the_scan_finds_the_purges_that_cover_a_file_as_each_purge_judges_it() {
        // Issue #708: the scan files the purges sent while it ran
        // by name and looks up the names that could cover a file,
        // instead of trying every target on every file under the
        // index lock. It must find exactly the targets whose own
        // rule covers the file, for keys with and without `#`
        // variants, digest-form bases, and sources with and
        // without a path.
        let keys = [
            "300x200/https://example.com/a.png",
            "300x200/https://example.com/a.png#avif",
            "300x200/https://example.com/a.png#avif#dpr2",
            "fit-in/0x0/filters:blur(2)/s3://bucket/dir/a.png#webp#w300",
            "sha256 0123abcd/https://example.com/a.png#jxl",
            "100x100/a.png",
            "a.png",
            "x/",
            "/#",
            "",
        ];
        let mut names: Vec<String> = ["g", "h", "/", "#", "a.png#avif", "elsewhere"]
            .map(String::from)
            .to_vec();
        for key in keys {
            for (at, _) in key.char_indices().chain([(key.len(), ' ')]) {
                names.push(key[..at].to_string());
                names.push(key[at..].to_string());
            }
        }
        names.sort();
        names.dedup();
        let mut purges = ScanPurges::default();
        let mut targets = Vec::new();
        for name in &names {
            for target in [
                PurgeTarget::Key(name.clone()),
                PurgeTarget::Group(name.clone()),
                PurgeTarget::Source(name.clone()),
            ] {
                let hold = purges.add(&target);
                let at = purges
                    .holds
                    .iter()
                    .position(|held| Arc::ptr_eq(held, &hold))
                    .unwrap();
                // A repeat shares the hold.
                assert!(Arc::ptr_eq(&purges.add(&target), &hold));
                targets.push((target, at));
            }
        }
        assert_eq!(purges.holds.len(), targets.len());
        assert_eq!(purges.sent, 2 * targets.len());
        let mut checked = 0;
        for key in keys {
            for groups in [
                Vec::new(),
                vec!["g".to_string()],
                vec!["g".to_string(), "h".to_string(), key.to_string()],
            ] {
                let mut found = purges.covering(key, &groups);
                found.sort_unstable();
                found.dedup();
                let mut expected: Vec<usize> = targets
                    .iter()
                    .filter(|(target, _)| target.covers(key, &groups))
                    .map(|(_, at)| *at)
                    .collect();
                expected.sort_unstable();
                assert_eq!(found, expected, "{key:?} in {groups:?}");
                checked += expected.len();
            }
        }
        assert!(checked > 100, "{checked}");
    }

    #[test]
    fn the_boot_scan_counts_the_files_it_deletes_and_those_it_cannot() {
        // Issue #708: the scan deleted files that do not verify,
        // copies under another key's name and the oldest entries
        // past the budget without a word, and a delete that
        // failed left the file with nothing said either. Each
        // file it deletes now counts in its report; each it
        // cannot delete stays, and counts there and in the tier's
        // errors; both are logged at a limited rate.
        let payload = [7u8; 200];
        let old: Vec<String> = (0..3).map(|i| format!("old/{i}")).collect();
        let file = serialize_entry(&old[0], &image(&payload), &test_entry_key()).len();
        let one = entry_charge(&old[0], file, &[]);
        // Three entries, a file that does not verify, and a copy
        // of the newest under another key's name, in a budget
        // that holds two entries.
        let left = |dir: &Path| {
            left_by_a_previous_run(dir, &old, &payload);
            std::fs::write(
                dir.join(format!("{}.bin", key_stem("bad"))),
                b"CRABDISK6\nnot an entry",
            )
            .unwrap();
            std::fs::copy(
                dir.join(format!("{}.bin", key_stem(&old[2]))),
                dir.join(format!("{}.bin", key_stem("copy"))),
            )
            .unwrap();
        };
        let budget = 2 * one + one / 2;
        let dir = temp_dir("scan-deletes");
        left(&dir);
        let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
        let report = store.scan_report();
        assert_eq!(report.phase, ScanPhase::Done);
        assert_eq!(
            (
                report.files,
                report.entries,
                report.deleted,
                report.undeleted
            ),
            (5, 2, 3, 0)
        );
        assert_eq!(report.skipped(), 0);
        assert_eq!(store.errors(), 0);
        assert_eq!(lru_order(&store), old[1..].to_vec());
        for key in [old[0].as_str(), "bad", "copy"] {
            assert!(!store.file_for(&key_stem(key)).exists(), "{key}");
        }
        let _ = std::fs::remove_dir_all(&dir);
        // A disk that refuses the deletes: the files stay, out of
        // the index, and each refusal is an error.
        let dir = temp_dir("scan-undeleted");
        left(&dir);
        let store = Arc::new(DiskStore::unscanned(&dir, budget, test_entry_key()));
        store.set_hold(Some(Arc::new(|kind, _| match kind {
            OpKind::Unlink => Err(std::io::Error::from_raw_os_error(30)),
            _ => Ok(()),
        })));
        store.begin_scan();
        let report = scan_ended(&store);
        assert_eq!(report.phase, ScanPhase::Done);
        assert_eq!(
            (
                report.files,
                report.entries,
                report.deleted,
                report.undeleted
            ),
            (5, 2, 0, 3)
        );
        assert_eq!(report.skipped(), 0);
        assert_eq!(store.errors(), 3);
        assert_eq!(lru_order(&store), old[1..].to_vec());
        for key in [old[0].as_str(), "bad", "copy"] {
            assert!(store.file_for(&key_stem(key)).exists(), "{key}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_delete_the_boot_scan_cannot_make_counts_with_the_inline_ones() {
        // Issue #708 counts a delete the boot scan cannot make
        // in its report and the tier's errors, and issue #709 an
        // inline one that fails in
        // `crabcrop_cache_disk_unlinks_failed_total`, which
        // counts the scan's too: either way the file stays on
        // disk, out of the index, until the next boot.
        let dir = temp_dir("scan-unlink-fails");
        let bad = dir.join(format!("{}.bin", key_stem("bad")));
        std::fs::write(&bad, b"CRABDISK6\nnot an entry").unwrap();
        let store = Arc::new(DiskStore::unscanned(&dir, 1_000_000, test_entry_key()));
        store.set_hold(Some(Arc::new(|kind, _| match kind {
            OpKind::Unlink => Err(std::io::Error::from_raw_os_error(30)),
            _ => Ok(()),
        })));
        store.begin_scan();
        let report = scan_ended(&store);
        assert_eq!(report.phase, ScanPhase::Done);
        assert_eq!((report.deleted, report.undeleted), (0, 1));
        assert_eq!(store.unlinks_failed(), 1);
        assert_eq!(store.errors(), 1);
        assert!(bad.exists());
        // A delete the scan makes counts in neither.
        let dir = temp_dir("scan-unlink-works");
        std::fs::write(
            dir.join(format!("{}.bin", key_stem("bad"))),
            b"CRABDISK6\nnot an entry",
        )
        .unwrap();
        let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
        assert_eq!(store.scan_report().deleted, 1);
        assert_eq!(store.unlinks_failed(), 0);
        assert_eq!(store.errors(), 0);
    }

    /// The index's bookkeeping as it was before issue #708: the
    /// stamps in a hash map, and each eviction a scan of them
    /// all for the smallest. Only what decides which entries
    /// stay and what each is charged.
    mod min_scan {
        use super::super::{ALLOCATION_OVERHEAD, entry_charge, with_groups};
        use crate::cache::{PurgeTarget, WriteStamp};
        use std::collections::HashMap;

        /// The record as it was, for its size.
        #[allow(dead_code)]
        pub struct Record {
            stem: String,
            file_bytes: usize,
            charge: usize,
            groups: Vec<String>,
            written: Option<WriteStamp>,
            id: u64,
            owes_shared: bool,
        }

        pub const RECORD_OVERHEAD: usize = 2
            * (size_of::<(String, Record)>() + size_of::<(String, u64)>())
            + 3 * ALLOCATION_OVERHEAD;

        pub struct Entry {
            pub file_bytes: usize,
            pub charge: usize,
            pub groups: Vec<String>,
        }

        #[derive(Default)]
        pub struct Index {
            max_bytes: usize,
            pub entries: HashMap<String, Entry>,
            stamps: HashMap<String, u64>,
            clock: u64,
            pub total_bytes: usize,
        }

        impl Index {
            pub fn new(max_bytes: usize) -> Self {
                Index {
                    max_bytes,
                    ..Default::default()
                }
            }

            /// The keys, from the least recently used.
            pub fn order(&self) -> Vec<String> {
                let mut stamped: Vec<(u64, &String)> = self
                    .stamps
                    .iter()
                    .map(|(key, stamp)| (*stamp, key))
                    .collect();
                stamped.sort();
                stamped.into_iter().map(|(_, key)| key.clone()).collect()
            }

            fn touch(&mut self, key: &str) {
                self.clock += 1;
                let stamp = self.clock;
                if let Some(s) = self.stamps.get_mut(key) {
                    *s = stamp;
                }
            }

            pub fn get(&mut self, key: &str) -> bool {
                if self.max_bytes == 0 || !self.entries.contains_key(key) {
                    return false;
                }
                self.touch(key);
                true
            }

            /// `None` for a key not held, else whether it grew.
            pub fn learn(&mut self, key: &str, groups: &[String]) -> Option<bool> {
                if self.max_bytes == 0 {
                    return None;
                }
                let entry = self.entries.get_mut(key)?;
                let merged = with_groups(&entry.groups, groups);
                let grew = merged.len() > entry.groups.len();
                if grew {
                    entry.groups = merged;
                    let charge = entry_charge(key, entry.file_bytes, &entry.groups);
                    self.total_bytes = self.total_bytes - entry.charge + charge;
                    entry.charge = charge;
                    self.clock += 1;
                    let stamp = self.clock;
                    if let Some(s) = self.stamps.get_mut(key) {
                        *s = stamp;
                    }
                }
                Some(grew)
            }

            pub fn remove(&mut self, key: &str) -> Option<Entry> {
                let entry = self.entries.remove(key)?;
                self.total_bytes -= entry.charge;
                self.stamps.remove(key);
                Some(entry)
            }

            /// [`super::super::DiskStore::put`] of a copy in
            /// `groups` whose file, in the groups it lands with,
            /// is `file_bytes` long. Returns how many entries it
            /// evicted.
            pub fn put(
                &mut self,
                key: &str,
                groups: &[String],
                file_bytes: impl Fn(&[String]) -> usize,
            ) -> usize {
                if self.max_bytes == 0 {
                    return 0;
                }
                let held = self
                    .remove(key)
                    .map(|entry| entry.groups)
                    .unwrap_or_default();
                let groups = with_groups(groups, &held);
                let file_bytes = file_bytes(&groups);
                let charge = entry_charge(key, file_bytes, &groups);
                if charge > self.max_bytes {
                    return 0;
                }
                let mut evicted = 0;
                while self.total_bytes + charge > self.max_bytes {
                    let oldest = self
                        .stamps
                        .iter()
                        .min_by_key(|(_, stamp)| *stamp)
                        .map(|(k, _)| k.clone());
                    let Some(oldest) = oldest else { break };
                    self.remove(&oldest);
                    evicted += 1;
                }
                self.total_bytes += charge;
                self.clock += 1;
                self.stamps.insert(key.to_string(), self.clock);
                self.entries.insert(
                    key.to_string(),
                    Entry {
                        file_bytes,
                        charge,
                        groups,
                    },
                );
                evicted
            }

            pub fn purge(&mut self, target: &PurgeTarget) -> usize {
                let covered: Vec<String> = self
                    .entries
                    .iter()
                    .filter(|(key, entry)| target.covers(key, &entry.groups))
                    .map(|(key, _)| key.clone())
                    .collect();
                for key in &covered {
                    self.remove(key);
                }
                covered.len()
            }
        }
    }

    #[test]
    fn an_index_record_is_charged_what_it_was_before_the_order_was_a_map() {
        // Issue #708: the record gained a stamp, and the charge
        // did not move with it. A budget holds the entries it
        // held, whatever their keys, groups and files.
        assert_eq!(RECORD_OVERHEAD, min_scan::RECORD_OVERHEAD);
    }

    #[test]
    fn the_index_evicts_what_the_min_scan_evicted_in_every_order_of_operations() {
        // Issue #708: eviction takes its victim from the front of
        // an ordered map and no longer reads every stamp to find
        // it. Whatever is done to the store (writes that replace
        // and writes that evict, hits, misses, learned groups that
        // grow an entry, removals and the three purges), it must
        // hold the same entries, in the same order of use, under
        // the same charges, and give every call the same answer,
        // as the bookkeeping that scanned.
        let ops = ["100x100", "200x0", "filters:blur(2)"];
        let sources = ["http://x/a.png", "http://x/b.png", "http://y/a.png"];
        let variants = ["", "#avif", "#webp#w300"];
        let mut keys = Vec::new();
        for op in ops {
            for source in sources {
                for variant in variants {
                    keys.push(format!("{op}/{source}{variant}"));
                }
            }
        }
        let names = ["thumb", "card", "hero"];
        let some_groups = |rng: &mut crate::test_support::Rng| -> Vec<String> {
            let mut some = Vec::new();
            for name in names {
                if rng.below(4) == 0 {
                    some.push(name.to_string());
                }
            }
            some
        };
        let one = entry_charge(
            &keys[0],
            serialize_entry(&keys[0], &image(&[7; 900]), &test_entry_key()).len(),
            &[],
        );
        let mut steps = 0;
        let mut evictions = 0;
        // A budget of about six entries, one of sixty, one that
        // holds everything, and none.
        for budget in [6 * one, 60 * one, usize::MAX / 4, 0] {
            for seed in 0..20 {
                let dir = temp_dir(&format!("min-scan-{budget}-{seed}"));
                let mut rng = crate::test_support::Rng::new(seed);
                let store = DiskStore::open(&dir, budget, test_entry_key()).unwrap();
                let mut model = min_scan::Index::new(budget);
                for step in 0..300 {
                    let key = rng.pick(&keys).clone();
                    let what = match rng.below(100) {
                        0..35 => {
                            let len = [10, 300, 900, 2_500, 7_000][rng.below(5)];
                            let copy = CachedImage {
                                groups: some_groups(&mut rng),
                                ..image(&vec![7; len])
                            };
                            let file_bytes = |groups: &[String]| {
                                let landed = CachedImage {
                                    groups: groups.to_vec(),
                                    ..copy.clone()
                                };
                                serialize_entry(&key, &landed, &test_entry_key()).len()
                            };
                            store.put(&key, &copy);
                            evictions += model.put(&key, &copy.groups, file_bytes);
                            "put"
                        }
                        35..65 => {
                            assert_eq!(store.get(&key).is_some(), model.get(&key));
                            "get"
                        }
                        65..77 => {
                            let more = some_groups(&mut rng);
                            let grew = match store.learn_groups(&key, &more, Bucket::Unknown) {
                                Learned::Indexed { grew, .. } => Some(grew),
                                _ => None,
                            };
                            assert_eq!(grew, model.learn(&key, &more));
                            "learn_groups"
                        }
                        77..85 => {
                            assert_eq!(store.remove(&key), model.remove(&key).is_some());
                            "remove"
                        }
                        85..90 => {
                            let base = key.split('#').next().unwrap().to_string();
                            let target = PurgeTarget::Key(base);
                            assert_eq!(store.purge(&target).0, model.purge(&target));
                            "purge key"
                        }
                        90..95 => {
                            let target = PurgeTarget::Group(rng.pick(&names).to_string());
                            assert_eq!(store.purge(&target).0, model.purge(&target));
                            "purge group"
                        }
                        _ => {
                            let target = PurgeTarget::Source(rng.pick(&sources).to_string());
                            assert_eq!(store.purge(&target).0, model.purge(&target));
                            "purge source"
                        }
                    };
                    let at = format!("budget {budget}, seed {seed}, step {step}: {what} {key}");
                    assert_eq!(lru_order(&store), model.order(), "{at}");
                    let inner = store.inner.lock().unwrap();
                    for (key, record) in &inner.entries {
                        let entry = &model.entries[key];
                        assert_eq!(
                            (record.charge, record.file_bytes, &record.groups),
                            (entry.charge, entry.file_bytes, &entry.groups),
                            "{at}: {key}"
                        );
                    }
                    assert_eq!(inner.total_bytes, model.total_bytes, "{at}");
                    assert!(inner.total_bytes <= budget, "{at}");
                    drop(inner);
                    steps += 1;
                }
                drop(store);
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
        assert_eq!(steps, 4 * 20 * 300);
        assert!(evictions > 1_000, "{evictions} evictions");
    }
}
