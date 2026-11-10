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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::cache::{CachedImage, PurgeCutoff, PurgeTarget, Tombstone, WriteStamp};
use crate::url::ImageFormat;

type HmacSha256 = Hmac<Sha256>;

/// An index record: the entry's file stem, its charge (see
/// [`entry_charge`]), and its cache groups.
type Record = (String, usize, Vec<String>);

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
    /// Last-access stamp per key (monotonic clock). Eviction
    /// removes the smallest stamp, the least recently used
    /// entry: the same order the VecDeque kept, at O(1) per
    /// access instead of a linear scan per hit.
    stamps: HashMap<String, u64>,
    clock: u64,
    /// The sum of the indexed entries' charges.
    total_bytes: usize,
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
    permits: std::sync::Arc<tokio::sync::Semaphore>,
    /// Test-only: run ahead of every entry read in `get`, so
    /// a test can hold a read the way a slow disk would. The
    /// store refuses FIFOs (issue #545), so a test cannot
    /// hold a read on one.
    #[cfg(test)]
    read_hold: Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>>,
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
const DISK_OP_PERMITS: usize = 32;

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
            inner: Mutex::new(Inner::default()),
            errors: std::sync::atomic::AtomicU64::new(0),
            permits: std::sync::Arc::new(tokio::sync::Semaphore::new(DISK_OP_PERMITS)),
            #[cfg(test)]
            read_hold: Mutex::new(None),
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

    /// Run one store operation on tokio's blocking pool
    /// and wait for its result (issue #347). The store is
    /// synchronous filesystem I/O; calling it from the
    /// request path directly would stall a runtime worker
    /// for the duration of the read, the write, or the
    /// purge walk. A permit bounds how many operations
    /// run at once and is acquired before the hop, so
    /// waiting operations hold no thread. `None` when the
    /// task died before reporting: the caller degrades to
    /// the operation's failure shape (a miss, a count of
    /// zero, a dropped write), the store's standing
    /// best-effort rule, with the failure counted through
    /// [`DiskStore::note_task_failure`].
    pub async fn blocking<T, F>(self: &std::sync::Arc<Self>, op: F) -> Option<T>
    where
        F: FnOnce(&DiskStore) -> T + Send + 'static,
        T: Send + 'static,
    {
        let permit = self.permits.clone().acquire_owned().await.ok()?;
        let store = self.clone();
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            op(&store)
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

    /// Index existing entry files, oldest first, so the LRU
    /// order approximates the previous run's. Files that do
    /// not parse are deleted on sight.
    fn scan(&self) -> Result<(), String> {
        let mut found: Vec<(std::time::SystemTime, String, String, usize, Vec<String>)> =
            Vec::new();
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
                Some((key, groups)) => {
                    let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    found.push((mtime, key, stem, meta.len() as usize, groups));
                }
                None => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        found.sort_by_key(|(mtime, _, _, _, _)| *mtime);
        let mut inner = self.inner.lock().unwrap();
        for (_, key, stem, size, groups) in found {
            let charge = entry_charge(&key, size, &groups);
            inner.total_bytes += charge;
            inner.clock += 1;
            let stamp = inner.clock;
            inner.stamps.insert(key.clone(), stamp);
            inner.entries.insert(key, (stem, charge, groups));
        }
        Ok(())
    }

    /// Run `hold` ahead of every entry read from now on, or
    /// stop with `None`.
    #[cfg(test)]
    pub(crate) fn set_read_hold(&self, hold: Option<std::sync::Arc<dyn Fn() + Send + Sync>>) {
        *self.read_hold.lock().unwrap() = hold;
    }

    #[cfg(test)]
    fn run_read_hold(&self) {
        // Cloned out first: the hook blocks, and the lock
        // must stay free for `set_read_hold`.
        let hold = self.read_hold.lock().unwrap().clone();
        if let Some(hold) = hold {
            hold();
        }
    }

    fn file_for(&self, stem: &str) -> PathBuf {
        self.dir.join(format!("{stem}.bin"))
    }

    /// Look up `key`. A corrupt or vanished file is a miss and
    /// is cleaned up, never served.
    pub fn get(&self, key: &str) -> Option<CachedImage> {
        if self.max_bytes == 0 {
            return None;
        }
        let stem = {
            let inner = self.inner.lock().unwrap();
            inner.entries.get(key)?.0.clone()
        };
        let path = self.file_for(&stem);
        #[cfg(test)]
        self.run_read_hold();
        let parsed =
            read_capped(&path, self.max_bytes).and_then(|data| parse_entry(&data, &self.entry_key));
        match parsed {
            Some((stored_key, image)) if stored_key == key => {
                let mut inner = self.inner.lock().unwrap();
                if inner.entries.contains_key(key) {
                    touch(&mut inner, key);
                }
                Some(image)
            }
            _ => {
                let delete = {
                    let mut inner = self.inner.lock().unwrap();
                    inner.entries.contains_key(key) && evict_entry(&mut inner, key).is_some()
                };
                if delete {
                    let _ = std::fs::remove_file(&path);
                }
                None
            }
        }
    }

    /// Store `image` under `key`, evicting least recently used
    /// entries until the new entry's charge fits the budget.
    pub fn put(&self, key: &str, image: &CachedImage) {
        if self.max_bytes == 0 {
            return;
        }
        let data = serialize_entry(key, image, &self.entry_key);
        let charge = entry_charge(key, data.len(), &image.groups);
        if charge > self.max_bytes {
            return;
        }
        let stem = key_stem(key);
        let tmp = self.dir.join(tmp_name(&stem));
        if let Err(e) = std::fs::write(&tmp, &data) {
            self.note_error("cannot write a cache entry", &e);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, self.file_for(&stem)) {
            self.note_error("cannot move a cache entry into place", &e);
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        let mut evicted: Vec<String> = Vec::new();
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some((_, old_size, _)) = inner.entries.remove(key) {
                inner.total_bytes -= old_size;
                inner.stamps.remove(key);
            }
            while inner.total_bytes + charge > self.max_bytes {
                let oldest = inner
                    .stamps
                    .iter()
                    .min_by_key(|(_, stamp)| *stamp)
                    .map(|(k, _)| k.clone());
                let Some(oldest) = oldest else { break };
                if let Some((old_stem, size, _)) = inner.entries.remove(&oldest) {
                    inner.total_bytes -= size;
                    evicted.push(old_stem);
                }
                inner.stamps.remove(&oldest);
            }
            inner.total_bytes += charge;
            inner.clock += 1;
            let stamp = inner.clock;
            inner.stamps.insert(key.to_string(), stamp);
            inner
                .entries
                .insert(key.to_string(), (stem, charge, image.groups.clone()));
        }
        for old_stem in evicted {
            let _ = std::fs::remove_file(self.file_for(&old_stem));
        }
    }

    /// Remove the single entry stored under `key`, if
    /// any: its index record and its file. Returns whether
    /// an entry was removed.
    pub fn remove(&self, key: &str) -> bool {
        let stem = {
            let mut inner = self.inner.lock().unwrap();
            evict_entry(&mut inner, key).map(|(stem, _)| stem)
        };
        match stem {
            Some(stem) => {
                let _ = std::fs::remove_file(self.file_for(&stem));
                true
            }
            None => false,
        }
    }

    /// Remove every entry for `base`: the bare key plus its
    /// `#`-suffixed variants (Accept class, DPR). Returns how
    /// many entries were removed.
    pub fn purge_matching(&self, base: &str) -> usize {
        let stems: Vec<String> = {
            let mut guard = self.inner.lock().unwrap();
            let inner = &mut *guard;
            // Built once, not per key examined.
            let prefix = format!("{base}#");
            inner
                .entries
                .extract_if(|k, _| k.as_str() == base || k.starts_with(&prefix))
                .map(|(key, (stem, size, _))| {
                    inner.total_bytes -= size;
                    inner.stamps.remove(&key);
                    stem
                })
                .collect()
        };
        for stem in &stems {
            let _ = std::fs::remove_file(self.file_for(stem));
        }
        stems.len()
    }

    /// Remove every entry belonging to `group` (RFC 9875
    /// grouped invalidation), using the groups indexed at
    /// scan and at store time. Returns how many entries
    /// were removed.
    pub fn purge_group(&self, group: &str) -> usize {
        let stems: Vec<String> = {
            let mut guard = self.inner.lock().unwrap();
            let inner = &mut *guard;
            inner
                .entries
                .extract_if(|_, (_, _, groups)| groups.iter().any(|g| g == group))
                .map(|(key, (stem, size, _))| {
                    inner.total_bytes -= size;
                    inner.stamps.remove(&key);
                    stem
                })
                .collect()
        };
        for stem in &stems {
            let _ = std::fs::remove_file(self.file_for(stem));
        }
        stems.len()
    }

    /// Union `groups` into the stored entry for `key`
    /// (audit R2-15): provenance is accumulated, not
    /// first-writer only, so a group purge can find an
    /// entry however it was first populated. The file is
    /// rewritten only when the union actually grows the
    /// set. Returns whether the stored entry changed.
    pub fn merge_groups(&self, key: &str, groups: &[String]) -> bool {
        let Some(mut image) = self.get(key) else {
            return false;
        };
        let merged = crate::cache::union_groups(&image.groups, groups);
        if merged == image.groups {
            return false;
        }
        image.groups = merged;
        self.put(key, &image);
        true
    }

    /// Remove every entry whose result derives from
    /// `source`, whatever operations produced it (audit
    /// R2-07): an upload overwrite calls this so the
    /// replaced original's derivatives stop serving.
    /// Returns how many entries were removed.
    pub fn purge_source(&self, source: &str) -> usize {
        let stems: Vec<String> = {
            let mut guard = self.inner.lock().unwrap();
            let inner = &mut *guard;
            // Built once, not per key examined.
            let suffix = format!("/{source}");
            inner
                .entries
                .extract_if(|k, _| {
                    let base = k.split('#').next().unwrap_or(k);
                    base == source || base.ends_with(&suffix)
                })
                .map(|(key, (stem, size, _))| {
                    inner.total_bytes -= size;
                    inner.stamps.remove(&key);
                    stem
                })
                .collect()
        };
        for stem in &stems {
            let _ = std::fs::remove_file(self.file_for(stem));
        }
        stems.len()
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

fn evict_entry(inner: &mut Inner, key: &str) -> Option<(String, usize)> {
    let (stem, size, _) = inner.entries.remove(key)?;
    inner.total_bytes -= size;
    inner.stamps.remove(key);
    Some((stem, size))
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
fn read_header(path: &Path, cap: usize, entry_key: &[u8]) -> Option<(String, Vec<String>)> {
    let data = read_capped(path, cap)?;
    let header = parse_header_bytes(&data, data.len())?;
    if !verify_entry_mac(entry_key, &data[header.covered_start..], &header.mac) {
        return None;
    }
    Some((header.key, header.groups))
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
    fn merge_groups_unions_provenance_and_persists() {
        let dir = temp_dir("merge");
        {
            let store = DiskStore::open(&dir, 1_000_000, test_entry_key()).unwrap();
            store.put("k", &grouped_image(b"x", &["thumb"]));
            // A later spelling brings a second group: the
            // stored set grows to the union (R2-15), and a
            // repeat merge changes nothing.
            assert!(store.merge_groups("k", &["hero".to_string()]));
            assert!(!store.merge_groups("k", &["thumb".to_string()]));
            assert_eq!(store.get("k").unwrap().groups, ["hero", "thumb"]);
            assert!(!store.merge_groups("missing", &["hero".to_string()]));
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
}
