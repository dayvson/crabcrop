//! Server-side result cache: processed responses keyed by the
//! verified request path.
//!
//! A cache key names a request (a transformation of a source
//! locator), not the bytes it produced: if the source changes,
//! a re-render under the same key yields different bytes. Each
//! entry therefore carries the strong ETag of its own bytes,
//! computed at render time, so conditional requests validate
//! against what was actually rendered. The cache is bounded
//! by total bytes with least-recently-used eviction, and an
//! entry is charged everything it keeps resident (its key,
//! its ETag and group names, and its bookkeeping, not only
//! its payload), so the budget bounds the memory the tier
//! holds. Only successful responses are stored; errors are
//! never cached.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::url::ImageFormat;

/// A stored response: everything needed to answer identically,
/// including the auto-quality header value when there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedImage {
    pub bytes: bytes::Bytes,
    pub format: ImageFormat,
    pub quality: Option<u8>,
    /// Cache groups (RFC 9875) the response belongs to: the
    /// preset names its URL referenced, empty for preset-free
    /// requests. Group purge matches on these.
    pub groups: Vec<String>,
    /// The strong ETag of `bytes`, computed when the render
    /// was produced and stored with the entry, so a cache hit
    /// answers conditional requests without rehashing the
    /// body. Because it derives from the bytes, a re-render
    /// under the same key whose bytes differ (the source
    /// changed, the old entry was purged) carries a different
    /// validator.
    pub etag: String,
    /// When and where the render behind these bytes began
    /// (issues #446, #461), persisted with every disk and
    /// shared copy and carried unchanged by promotions and
    /// group merges, so a copy's age survives every tier it
    /// passes through. `None` for an entry written before
    /// stamps existed: such a copy counts as older than every
    /// purge.
    pub written: Option<WriteStamp>,
}

/// Whether two images are the same stored copy: the same
/// bytes from the same render. A re-render of unchanged
/// bytes has the same ETag but a later stamp, and a purge
/// reconcile must not mistake it for the copy it removes.
pub(crate) fn same_copy(a: &CachedImage, b: &CachedImage) -> bool {
    a.etag == b.etag && a.written == b.written
}

/// The wall clock in Unix milliseconds: the time base that
/// write stamps and purge starts share across replicas.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// When the render behind a copy began, and which process
/// ran it (issues #446, #461). The time is taken when the
/// render starts, before it reads its source, so a copy
/// stamped after a purge began cannot hold bytes from
/// before it. The writer is a per-boot process id: a copy
/// and a purge from the same process share one clock and
/// compare exactly, while a copy from any other process is
/// compared with a clock-skew margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteStamp {
    pub at_ms: u64,
    pub writer: u64,
}

/// How far another process's clock may run ahead of the one
/// that started a purge before a copy it rendered just
/// before the purge escapes it (issue #461). A copy from
/// another process counts as written before the purge until
/// this long after the purge began; NTP-synced hosts agree
/// to within milliseconds. The cost of the margin is that a
/// copy another process renders in the first seconds after
/// a purge counts as purged: it is not cached, and the next
/// request for it renders again.
pub const CLOCK_SKEW_MARGIN_MS: u64 = 5_000;

/// How long a purge's tombstone outlives the end of its
/// shared-tier walk (issue #461). After the walk, the bucket
/// holds no copy the purge covers, except ones written back
/// by renders that were already running somewhere when the
/// purge began; the grace keeps those missing while they
/// land, so it sits well above the longest render plus the
/// tombstone refresh interval.
pub const TOMBSTONE_GRACE_MS: u64 = 15 * 60 * 1000;

/// When a purge began, and on which process: the line
/// between the copies it removes and the ones rendered
/// after it (issues #446, #461).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PurgeCutoff {
    pub start_ms: u64,
    pub writer: u64,
}

impl PurgeCutoff {
    /// A cutoff that covers every copy, stamped or not: the
    /// unbounded purge forms, which remove everything they
    /// match.
    pub const EVERYTHING: PurgeCutoff = PurgeCutoff {
        start_ms: u64::MAX,
        writer: 0,
    };

    /// Whether a copy stamped `written` was rendered before
    /// this purge began. An unstamped copy predates every
    /// purge. A copy from the purging process compares
    /// exactly; one from any other process is covered until
    /// [`CLOCK_SKEW_MARGIN_MS`] after the purge began, so a
    /// writer whose clock runs ahead cannot slip a pre-purge
    /// copy past it.
    pub fn covers(&self, written: Option<WriteStamp>) -> bool {
        match written {
            None => true,
            Some(stamp) if stamp.writer == self.writer => stamp.at_ms < self.start_ms,
            Some(stamp) => stamp.at_ms < self.start_ms.saturating_add(CLOCK_SKEW_MARGIN_MS),
        }
    }

    /// Whether every copy `other` covers, this cutoff covers
    /// too. Same process: a start no earlier. Different
    /// processes: a start at least the margin later, since
    /// each one covers the other's copies with the margin.
    fn includes(&self, other: &PurgeCutoff) -> bool {
        if self.writer == other.writer {
            other.start_ms <= self.start_ms
        } else {
            other.start_ms.saturating_add(CLOCK_SKEW_MARGIN_MS) <= self.start_ms
        }
    }
}

/// One purge as every replica sees it (issue #461): what it
/// removes, when it began, and when its shared-tier walk
/// ended. The process that starts a purge holds one in
/// memory and writes it to the shared bucket, where the
/// other replicas, and the same one after a restart, read
/// it. While it lives, a disk or shared copy it covers that
/// was written before the purge began is a miss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tombstone {
    pub target: PurgeTarget,
    pub cutoff: PurgeCutoff,
    /// When the purge's walk ended; `None` while it runs, or
    /// for a walk that never finished (its process stopped).
    pub ended_ms: Option<u64>,
}

impl Tombstone {
    /// Whether the tombstone still applies at `now_ms`: until
    /// [`TOMBSTONE_GRACE_MS`] after its walk ended, and
    /// indefinitely for a walk that has not.
    pub fn live(&self, now_ms: u64) -> bool {
        self.ended_ms
            .is_none_or(|ended| now_ms < ended.saturating_add(TOMBSTONE_GRACE_MS))
    }

    /// Whether the purge removes a copy stored under `key`
    /// with `groups`, stamped `written`.
    pub fn covers(&self, key: &str, groups: &[String], written: Option<WriteStamp>) -> bool {
        self.target.covers(key, groups) && self.cutoff.covers(written)
    }

    /// Whether this tombstone makes `other` redundant: same
    /// target, a walk that has ended, and a cutoff that covers
    /// strictly more. The finished walk removed every copy
    /// `other` covers, so `other` can go; this is how a
    /// re-purge clears the tombstone of a walk whose process
    /// stopped before it ended.
    pub fn supersedes(&self, other: &Tombstone) -> bool {
        self.ended_ms.is_some()
            && self.target == other.target
            && self.cutoff != other.cutoff
            && self.cutoff.includes(&other.cutoff)
    }
}

/// The strong ETag for a body of rendered bytes: a quoted
/// SHA-256 digest truncated to 128 bits, the shape the
/// validators have always had. It is computed over the bytes
/// themselves, never over the cache key: a key names a
/// transformation of a source locator, and the source can
/// change under it.
pub(crate) fn etag_for_bytes(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(bytes);
    format!("\"{}\"", hex::encode(&digest[..16]))
}

/// Whether a cache key's result derives from `source`.
/// The key is the canonical form (operations, then the
/// source) plus any `#`-suffixed variant segments
/// (negotiated class, request hints, the config
/// fingerprint), and a source string itself never
/// contains a `#` (it is a URL remainder, where `#` would
/// start a fragment), so the base is the key up to the
/// first `#` and the source is its tail segment.
///
/// Matched in place, with no `/{source}` string built per
/// call (issue #462): a running purge's tombstone runs this
/// on every disk and shared hit.
pub(crate) fn key_belongs_to_source(key: &str, source: &str) -> bool {
    let base = key.split('#').next().unwrap_or(key);
    base.strip_suffix(source)
        .is_some_and(|head| head.is_empty() || head.ends_with('/'))
}

/// The longest cache key base (the canonical form, ahead of
/// the `#` suffixes) a key carries as written. Real request
/// paths sit far below it. A longer base, which only a
/// padded source or an extreme operation chain produces, is
/// carried in digest form instead (see [`bounded_key_base`]),
/// so one entry's key, which every tier holds and the memory
/// tiers hold twice, stays small next to the budgets.
pub(crate) const MAX_KEY_BASE_BYTES: usize = 4096;

/// What opens a digest-form key base. It holds a space, which
/// a request target cannot carry, so no base a request spells
/// out can begin with it and the two forms never meet.
const DIGEST_BASE_MARK: &str = "sha256 ";

/// The cache key base for `base`, the canonical form of a
/// request whose source is `source`. A base within
/// [`MAX_KEY_BASE_BYTES`] is returned unchanged, so the keys of
/// ordinary URLs never change. A longer one becomes the
/// SHA-256 of the whole base, followed by `/` and the source
/// when the source is the base's tail and the result still
/// fits: keeping the source keeps the key under the
/// per-source purge an upload overwrite runs
/// ([`key_belongs_to_source`]). The serve path and the purge
/// path both derive the base through here, so a purge by URL
/// still finds an entry keyed in digest form.
pub(crate) fn bounded_key_base(base: String, source: &str) -> String {
    if base.len() <= MAX_KEY_BASE_BYTES {
        return base;
    }
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(base.as_bytes());
    let mut bounded = format!("{DIGEST_BASE_MARK}{}", hex::encode(digest));
    let keeps_source = !source.is_empty()
        && base.ends_with(source)
        && bounded.len() + 1 + source.len() <= MAX_KEY_BASE_BYTES;
    if keeps_source {
        bounded.push('/');
        bounded.push_str(source);
    }
    bounded
}

/// Whether `key` is the base cache key `base` or one of its
/// `#`-suffixed variants: the rule a per-URL purge removes
/// by. Matched in place, with no `{base}#` prefix built per
/// call (issue #462).
fn key_has_base(key: &str, base: &str) -> bool {
    key.strip_prefix(base)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('#'))
}

/// The union of two group sets, sorted so every tier
/// stores and compares the same canonical set.
pub(crate) fn union_groups(existing: &[String], extra: &[String]) -> Vec<String> {
    let mut out: Vec<String> = existing.to_vec();
    for group in extra {
        if !out.contains(group) {
            out.push(group.clone());
        }
    }
    out.sort();
    out
}

/// How many generation tombstones of one kind the guard
/// retains, and how many ended purges it keeps within
/// their grace. Purges are operator actions, so the
/// distinct targets stay few in practice; the cap only
/// bounds the pathological, and a write old enough to
/// outlive this many later purges is far outside any
/// render's lifetime (renders are bounded by the process
/// timeout, not by operator patience).
const MAX_TOMBSTONES: usize = 1024;

/// Purge generations and tombstones (issue #350): the
/// coordination that keeps a cache write begun before a
/// purge from restoring what the purge removed.
///
/// A writer (a render populating the tiers, a hit
/// promoting or merging an entry) takes a `snapshot` of
/// the generation when it starts. A purge records a
/// tombstone for its target at a fresh generation before
/// it touches any tier. A write whose snapshot predates
/// a tombstone covering its key or groups is stale: it
/// must not land, and if it already landed (the purge
/// arrived while the write was in flight) it is removed
/// again. A write that starts after the purge snapshots
/// the newer generation and is unaffected, so the guard
/// costs a fresh render nothing.
///
/// The guard also decides which persisted copies a purge
/// still removes after its request (issues #400, #446,
/// #461). A shared-tier purge walk can outlive its request
/// by minutes on a large bucket, and until it reaches an
/// object that object still reads as a hit; a lookup that
/// starts after the purge snapshots the newer generation,
/// so the generation tombstone alone would let it serve
/// that copy and promote it into the local tiers, where
/// the walk never looks. So every purge also leaves a
/// [`Tombstone`]: from the moment it begins until a grace
/// period after its walk ends, a disk or shared copy it
/// covers that was written before it began is a miss, and
/// is removed on sight. A copy rendered after it began is
/// not: it is a valid render and serves normally, so a
/// running walk costs no re-renders. The guard holds the
/// tombstones of the purges this process started and the
/// ones read from the shared bucket, which is how a purge
/// reaches other replicas and survives a restart.
///
/// The guard tracks purges, not entries: it holds no
/// image data and its maps stay proportional to how many
/// distinct targets were purged, capped above, plus the
/// purges still running and the live tombstones, also
/// capped.
#[derive(Debug)]
pub struct PurgeGuard {
    inner: Mutex<PurgeGuardInner>,
    /// This process's writer id for [`WriteStamp`]s and
    /// purge cutoffs: random per boot, so a restarted
    /// process compares its new copies with the margin
    /// against its own pre-restart purges.
    writer: u64,
    /// The latest time this process has issued as a stamp
    /// or a purge start. Copies and purges of one process
    /// compare with no margin, so their times must follow
    /// the order of events even if the host clock steps
    /// back: issued times never decrease, and a purge
    /// starts strictly after every stamp issued before it.
    clock: AtomicU64,
}

impl Default for PurgeGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default)]
struct PurgeGuardInner {
    /// Bumped by every recorded purge; a snapshot is a
    /// reading of this counter.
    generation: u64,
    /// Base cache key -> generation of its purge. Matches
    /// the key itself and its `#`-suffixed variants, the
    /// same rule `Cache::purge_matching` removes by.
    keys: HashMap<String, u64>,
    /// Group name -> generation of its purge.
    groups: HashMap<String, u64>,
    /// Source -> generation of its purge. Matches the
    /// keys that derive from the source, the rule
    /// `Cache::purge_source` removes by.
    sources: HashMap<String, u64>,
    /// Tombstones of the purges this process started, by
    /// generation: the running ones (one per live
    /// [`ActivePurge`]) and the ended ones still within
    /// their grace. Running ones are never trimmed; ended
    /// ones are capped at [`MAX_TOMBSTONES`].
    local: HashMap<u64, Tombstone>,
    /// Tombstones read from the shared bucket (issue #461),
    /// by object name, replaced whole by each refresh.
    shared: HashMap<String, Tombstone>,
}

/// What one purge removes: the matching rule the tiers
/// remove by, kept for as long as the purge runs. Two
/// purges with equal targets are identical apart from when
/// they began, which is how the shared tier coalesces their
/// walks (issue #462).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PurgeTarget {
    /// A base cache key: the bare key and its
    /// `#`-suffixed variants.
    Key(String),
    /// Every entry recorded in the cache group.
    Group(String),
    /// Every entry derived from the source.
    Source(String),
}

impl PurgeTarget {
    /// Whether this purge removes an entry stored under
    /// `key` with `groups`.
    pub(crate) fn covers(&self, key: &str, groups: &[String]) -> bool {
        match self {
            PurgeTarget::Key(base) => key_has_base(key, base),
            PurgeTarget::Group(group) => groups.iter().any(|g| g == group),
            PurgeTarget::Source(source) => key_belongs_to_source(key, source),
        }
    }
}

/// A purge in progress (issue #400), from its tombstone
/// until the last tier it reaches is done, the shared
/// walk's background remainder included. Dropping the
/// handle ends it, so a purge task that finishes, panics,
/// or is torn down never leaves it behind; its tombstone
/// then lives on for [`TOMBSTONE_GRACE_MS`].
#[must_use = "the purge counts as active only while the handle lives"]
#[derive(Debug)]
pub struct ActivePurge {
    guard: Arc<PurgeGuard>,
    generation: u64,
    tombstone: Tombstone,
}

impl ActivePurge {
    /// The purge's tombstone as it stands while it runs:
    /// what the shared bucket records for the other
    /// replicas (issue #461).
    pub fn tombstone(&self) -> &Tombstone {
        &self.tombstone
    }

    /// The purge's generation, unique within this process:
    /// part of its tombstone's object name.
    pub fn sequence(&self) -> u64 {
        self.generation
    }
}

impl Drop for ActivePurge {
    fn drop(&mut self) {
        let mut inner = self
            .guard
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = now_ms();
        if let Some(tombstone) = inner.local.get_mut(&self.generation) {
            tombstone.ended_ms = Some(now);
        }
        trim_local(&mut inner.local, now);
    }
}

/// Drop the local tombstones whose grace has passed, then
/// the oldest ended ones past the cap. Running purges stay.
fn trim_local(local: &mut HashMap<u64, Tombstone>, now: u64) {
    local.retain(|_, tombstone| tombstone.live(now));
    let mut ended: Vec<u64> = local
        .iter()
        .filter(|(_, tombstone)| tombstone.ended_ms.is_some())
        .map(|(generation, _)| *generation)
        .collect();
    if ended.len() > MAX_TOMBSTONES {
        ended.sort_unstable();
        for generation in &ended[..ended.len() - MAX_TOMBSTONES] {
            local.remove(generation);
        }
    }
}

impl PurgeGuard {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PurgeGuardInner::default()),
            writer: boot_writer_id(),
            clock: AtomicU64::new(0),
        }
    }

    /// A stamp time for `now`: never earlier than any time
    /// this process issued before.
    fn stamp_ms(&self, now: u64) -> u64 {
        self.clock.fetch_max(now, Ordering::SeqCst).max(now)
    }

    /// A purge start for `now`: strictly later than every
    /// stamp this process issued before it, so a copy
    /// rendered before the purge is covered by it even
    /// within the same millisecond or after the host clock
    /// stepped back, and every stamp issued after it is no
    /// earlier.
    fn purge_start_ms(&self, now: u64) -> u64 {
        let next = |last: u64| last.saturating_add(1).max(now);
        // A plain compare-exchange loop rather than
        // fetch_update, which newer toolchains deprecate in
        // favour of try_update: the loop builds the same on
        // every toolchain from the MSRV up.
        let mut last = self.clock.load(Ordering::SeqCst);
        loop {
            match self.clock.compare_exchange_weak(
                last,
                next(last),
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return next(last),
                Err(actual) => last = actual,
            }
        }
    }

    /// The current purge generation: a writer snapshots
    /// this when it starts, before it reads or renders
    /// anything it might later write back.
    pub fn snapshot(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    /// The stamp for a render starting now on this process
    /// (issues #446, #461). Taken after the render's
    /// [`PurgeGuard::snapshot`], so a purge that begins
    /// between the two is caught by the generation, and one
    /// that begins after both starts strictly later than the
    /// stamp, so it covers the render's copies.
    pub fn stamp(&self) -> WriteStamp {
        WriteStamp {
            at_ms: self.stamp_ms(now_ms()),
            writer: self.writer,
        }
    }

    /// Start a purge of `target`: record its generation
    /// tombstone and its stamped tombstone in one step, so
    /// no lookup or write can fall between the two. The
    /// purge stays running until the returned handle is
    /// dropped; the caller holds it until every tier it
    /// purges is done.
    pub fn begin(self: &Arc<Self>, target: PurgeTarget) -> ActivePurge {
        let mut inner = self.inner.lock().unwrap();
        let generation = record_tombstone(&mut inner, &target);
        let tombstone = Tombstone {
            target,
            cutoff: PurgeCutoff {
                start_ms: self.purge_start_ms(now_ms()),
                writer: self.writer,
            },
            ended_ms: None,
        };
        inner.local.insert(generation, tombstone.clone());
        ActivePurge {
            guard: Arc::clone(self),
            generation,
            tombstone,
        }
    }

    /// Replace the tombstones read from the shared bucket
    /// with a fresh reading (issue #461).
    pub fn replace_shared(&self, tombstones: HashMap<String, Tombstone>) {
        self.inner.lock().unwrap().shared = tombstones;
    }

    /// Record a purge of the entries keyed at `base` (the
    /// bare key and its variants), tombstone only.
    /// Returns the purge's generation.
    #[cfg(test)]
    pub fn record_key(&self, base: &str) -> u64 {
        let target = PurgeTarget::Key(base.to_string());
        record_tombstone(&mut self.inner.lock().unwrap(), &target)
    }

    /// Record a purge of `group`'s member entries,
    /// tombstone only. Returns the purge's generation.
    #[cfg(test)]
    pub fn record_group(&self, group: &str) -> u64 {
        let target = PurgeTarget::Group(group.to_string());
        record_tombstone(&mut self.inner.lock().unwrap(), &target)
    }

    /// Record a purge of every entry derived from
    /// `source`, tombstone only. Returns the purge's
    /// generation.
    #[cfg(test)]
    pub fn record_source(&self, source: &str) -> u64 {
        let target = PurgeTarget::Source(source.to_string());
        record_tombstone(&mut self.inner.lock().unwrap(), &target)
    }

    /// Whether a live tombstone covers a persisted copy
    /// stored under `key` with `groups` and stamped
    /// `written` (issues #400, #446, #461): a purge this
    /// process started that is still running or within its
    /// grace, or one read from the shared bucket, that began
    /// after the copy was written. Such a copy is a miss,
    /// whenever the lookup that found it started: the purge
    /// may not have reached it yet. A copy rendered after
    /// every matching purge began is not covered.
    pub fn running_covers(
        &self,
        key: &str,
        groups: &[String],
        written: Option<WriteStamp>,
    ) -> bool {
        self.running_covers_at(now_ms(), key, groups, written)
    }

    /// [`PurgeGuard::running_covers`] at a given time.
    pub(crate) fn running_covers_at(
        &self,
        now: u64,
        key: &str,
        groups: &[String],
        written: Option<WriteStamp>,
    ) -> bool {
        let inner = self.inner.lock().unwrap();
        inner
            .local
            .values()
            .chain(inner.shared.values())
            .any(|tombstone| tombstone.live(now) && tombstone.covers(key, groups, written))
    }

    /// Whether a write of a copy stamped `written`, begun
    /// under `snapshot`, must not stand: a purge recorded
    /// after the snapshot covers it (issue #350), or a live
    /// tombstone covers the copy itself (issue #461), as
    /// when a replica writes back a copy it held from before
    /// another replica's purge.
    pub fn write_covered(
        &self,
        snapshot: u64,
        key: &str,
        groups: &[String],
        written: Option<WriteStamp>,
    ) -> bool {
        self.purge_covers(snapshot, key, groups) || self.running_covers(key, groups, written)
    }

    /// How many purges are still running.
    #[cfg(test)]
    pub fn active_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap()
            .local
            .values()
            .filter(|tombstone| tombstone.ended_ms.is_none())
            .count()
    }

    /// Whether a purge recorded after `snapshot` covers
    /// an entry stored under `key` with `groups`: such a
    /// write began before the purge and must not stand.
    pub fn purge_covers(&self, snapshot: u64, key: &str, groups: &[String]) -> bool {
        let inner = self.inner.lock().unwrap();
        inner
            .keys
            .iter()
            .any(|(base, purge_gen)| *purge_gen > snapshot && key_has_base(key, base))
            || inner.groups.iter().any(|(group, purge_gen)| {
                *purge_gen > snapshot && groups.iter().any(|g| g == group)
            })
            || inner.sources.iter().any(|(source, purge_gen)| {
                *purge_gen > snapshot && key_belongs_to_source(key, source)
            })
    }
}

/// A per-boot process id for write stamps: random, from
/// the standard library's per-process hash keys (seeded
/// from the OS), mixed with the clock and the pid. It only
/// has to tell processes apart; it guards nothing.
fn boot_writer_id() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    hasher.write_u32(std::process::id());
    hasher.finish()
}

/// Record `target`'s tombstone at a fresh generation
/// and return the generation.
fn record_tombstone(inner: &mut PurgeGuardInner, target: &PurgeTarget) -> u64 {
    inner.generation += 1;
    let generation = inner.generation;
    let (map, name) = match target {
        PurgeTarget::Key(base) => (&mut inner.keys, base),
        PurgeTarget::Group(group) => (&mut inner.groups, group),
        PurgeTarget::Source(source) => (&mut inner.sources, source),
    };
    map.insert(name.clone(), generation);
    trim_tombstones(map);
    generation
}

/// Drop the oldest tombstones until the map is back at
/// the cap: the ones a still-running write could
/// predate are the recent ones.
fn trim_tombstones(map: &mut HashMap<String, u64>) {
    while map.len() > MAX_TOMBSTONES {
        let Some(oldest) = map
            .iter()
            .min_by_key(|(_, purge_gen)| *purge_gen)
            .map(|(target, _)| target.clone())
        else {
            break;
        };
        map.remove(&oldest);
    }
}

/// One stored entry and what it is charged against the
/// budget. The charge is recorded with the entry, so every
/// removal takes back exactly what the store added, even
/// after a group merge has grown the entry.
#[derive(Debug)]
struct Entry {
    image: CachedImage,
    cost: usize,
}

/// The allocator's bookkeeping charged for each separate heap
/// allocation an entry makes: its header and size rounding.
const ALLOCATION_OVERHEAD: usize = 16;

/// What an entry costs besides its strings and its payload:
/// its slot in each of the two maps, doubled for the spare
/// slots a hash map keeps as it grows, and the allocations
/// behind its two key copies, its ETag, and its payload.
const ENTRY_OVERHEAD: usize =
    2 * (size_of::<(String, Entry)>() + size_of::<(String, u64)>()) + 4 * ALLOCATION_OVERHEAD;

/// What one entry keeps resident, charged against the budget:
/// the payload, the key twice (it keys both the entry map and
/// the recency map), the ETag, every group name with its own
/// string and allocation, and [`ENTRY_OVERHEAD`]. Charging only
/// the payload let entries with tiny bodies under long keys
/// hold many times the budget without ever being evicted.
fn entry_cost(key: &str, image: &CachedImage) -> usize {
    let groups: usize = image
        .groups
        .iter()
        .map(|group| group.len() + size_of::<String>() + ALLOCATION_OVERHEAD)
        .sum();
    image.bytes.len() + 2 * key.len() + image.etag.len() + groups + ENTRY_OVERHEAD
}

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<String, Entry>,
    /// Last-access stamp per key (monotonic clock). Eviction
    /// removes the smallest stamp, which is the least recently
    /// used entry: the same order the VecDeque kept, at O(1)
    /// per access instead of a linear scan per hit.
    stamps: HashMap<String, u64>,
    clock: u64,
    /// The sum of the stored entries' charges.
    total_bytes: usize,
}

/// A byte-bounded LRU cache: the entries' charges (see
/// [`entry_cost`]) stay within `max_bytes`. A `max_bytes` of 0
/// disables it.
#[derive(Debug)]
pub struct Cache {
    max_bytes: usize,
    inner: Mutex<Inner>,
}

impl Cache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Look up `key`, marking it most recently used.
    pub fn get(&self, key: &str) -> Option<CachedImage> {
        if self.max_bytes == 0 {
            return None;
        }
        let mut inner = self.inner.lock().unwrap();
        let image = inner.entries.get(key)?.image.clone();
        touch(&mut inner, key);
        Some(image)
    }

    /// Remove every entry equal to `base` or carrying it as a
    /// `#`-suffixed prefix (the cache's variant keys). Returns
    /// how many entries were removed.
    pub fn purge_matching(&self, base: &str) -> usize {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        // Built once, not per key examined.
        let prefix = format!("{base}#");
        let mut removed = 0;
        for (key, old) in inner
            .entries
            .extract_if(|k, _| k.as_str() == base || k.starts_with(&prefix))
        {
            inner.total_bytes -= old.cost;
            inner.stamps.remove(&key);
            removed += 1;
        }
        removed
    }

    /// Remove the single entry stored under `key`, if
    /// any. Returns whether an entry was removed.
    pub fn remove(&self, key: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(old) = inner.entries.remove(key) else {
            return false;
        };
        inner.total_bytes -= old.cost;
        inner.stamps.remove(key);
        true
    }

    /// Remove every entry belonging to `group` (RFC 9875
    /// grouped invalidation). Returns how many entries were
    /// removed.
    pub fn purge_group(&self, group: &str) -> usize {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        let mut removed = 0;
        for (key, old) in inner
            .entries
            .extract_if(|_, entry| entry.image.groups.iter().any(|g| g == group))
        {
            inner.total_bytes -= old.cost;
            inner.stamps.remove(&key);
            removed += 1;
        }
        removed
    }

    /// Union `groups` into the stored entry for `key`
    /// (audit R2-15): provenance is accumulated on every
    /// hit, so an entry first stored by a group-free
    /// spelling is still found by a later group purge.
    /// Returns whether the stored entry changed.
    ///
    /// The entry is charged again for its grown group set, and
    /// least recently used entries make room if that takes the
    /// cache over its budget.
    pub fn merge_groups(&self, key: &str, groups: &[String]) -> bool {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        let Some(entry) = inner.entries.get_mut(key) else {
            return false;
        };
        let merged = union_groups(&entry.image.groups, groups);
        if merged == entry.image.groups {
            return false;
        }
        entry.image.groups = merged;
        let cost = entry_cost(key, &entry.image);
        let old = std::mem::replace(&mut entry.cost, cost);
        inner.total_bytes = inner.total_bytes - old + cost;
        evict_to_fit(inner, self.max_bytes, 0);
        true
    }

    /// Remove every entry whose result derives from
    /// `source`, whatever operations produced it (audit
    /// R2-07): an upload overwrite calls this so the
    /// replaced original's derivatives stop serving.
    /// Returns how many entries were removed.
    pub fn purge_source(&self, source: &str) -> usize {
        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        // Built once, not per key examined.
        let suffix = format!("/{source}");
        let mut removed = 0;
        for (key, old) in inner.entries.extract_if(|k, _| {
            let base = k.split('#').next().unwrap_or(k);
            base == source || base.ends_with(&suffix)
        }) {
            inner.total_bytes -= old.cost;
            inner.stamps.remove(&key);
            removed += 1;
        }
        removed
    }

    /// Store `image` under `key`, evicting least recently used
    /// entries until its charge fits. Entries whose charge is
    /// larger than the whole budget are not stored.
    pub fn put(&self, key: String, image: CachedImage) {
        let cost = entry_cost(&key, &image);
        if self.max_bytes == 0 || cost > self.max_bytes {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        if let Some(old) = inner.entries.remove(&key) {
            inner.total_bytes -= old.cost;
            inner.stamps.remove(&key);
        }
        evict_to_fit(&mut inner, self.max_bytes, cost);
        inner.total_bytes += cost;
        inner.clock += 1;
        let stamp = inner.clock;
        inner.stamps.insert(key.clone(), stamp);
        inner.entries.insert(key, Entry { image, cost });
    }

    /// The stored entries' total charge.
    #[cfg(test)]
    pub(crate) fn charged_bytes(&self) -> usize {
        self.inner.lock().unwrap().total_bytes
    }
}

/// Evict least recently used entries until `incoming` more
/// bytes of charge fit within `max_bytes`.
fn evict_to_fit(inner: &mut Inner, max_bytes: usize, incoming: usize) {
    while inner.total_bytes + incoming > max_bytes {
        let oldest = inner
            .stamps
            .iter()
            .min_by_key(|(_, stamp)| *stamp)
            .map(|(k, _)| k.clone());
        let Some(oldest) = oldest else { break };
        if let Some(evicted) = inner.entries.remove(&oldest) {
            inner.total_bytes -= evicted.cost;
        }
        inner.stamps.remove(&oldest);
    }
}

fn touch(inner: &mut Inner, key: &str) {
    inner.clock += 1;
    let stamp = inner.clock;
    if let Some(s) = inner.stamps.get_mut(key) {
        *s = stamp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(n: usize) -> CachedImage {
        let bytes = vec![7; n];
        CachedImage {
            etag: etag_for_bytes(&bytes),
            bytes: bytes::Bytes::from(bytes),
            format: ImageFormat::Png,
            quality: None,
            groups: Vec::new(),
            written: None,
        }
    }

    fn grouped(n: usize, groups: &[&str]) -> CachedImage {
        CachedImage {
            groups: groups.iter().map(|g| g.to_string()).collect(),
            ..img(n)
        }
    }

    /// What `img(n)` stored under `key` is charged.
    fn cost(key: &str, n: usize) -> usize {
        entry_cost(key, &img(n))
    }

    /// A budget roomy enough that the tests using it never evict.
    const ROOMY: usize = 64 * 1024;

    #[test]
    fn etag_for_bytes_is_a_stable_quoted_digest_of_the_bytes() {
        let etag = etag_for_bytes(b"some rendered bytes");
        assert!(etag.starts_with('"') && etag.ends_with('"'));
        assert_eq!(etag.len(), 34);
        assert_eq!(etag, etag_for_bytes(b"some rendered bytes"));
        assert_ne!(etag, etag_for_bytes(b"other rendered bytes"));
    }

    #[test]
    fn put_then_get_round_trips() {
        let cache = Cache::new(ROOMY);
        cache.put("a".to_string(), img(10));
        assert_eq!(cache.get("a"), Some(img(10)));
        assert_eq!(cache.get("missing"), None);
    }

    #[test]
    fn eviction_drops_the_oldest_when_over_budget() {
        let cache = Cache::new(cost("a", 60) + cost("b", 60) - 1);
        cache.put("a".to_string(), img(60));
        cache.put("b".to_string(), img(60));
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), Some(img(60)));
    }

    #[test]
    fn get_refreshes_recency() {
        let cache = Cache::new(cost("a", 60) + cost("b", 30) + cost("c", 30) - 1);
        cache.put("a".to_string(), img(60));
        cache.put("b".to_string(), img(30));
        assert!(cache.get("a").is_some());
        // Over budget: b is now the least recently used, not a.
        cache.put("c".to_string(), img(30));
        assert_eq!(cache.get("b"), None);
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn oversized_entries_are_not_stored() {
        let cache = Cache::new(10);
        cache.put("big".to_string(), img(20));
        assert_eq!(cache.get("big"), None);
    }

    #[test]
    fn zero_budget_disables() {
        let cache = Cache::new(0);
        cache.put("a".to_string(), img(1));
        assert_eq!(cache.get("a"), None);
    }

    #[test]
    fn purge_group_removes_only_members() {
        let cache = Cache::new(ROOMY);
        cache.put("a".to_string(), grouped(10, &["thumb"]));
        cache.put("b".to_string(), grouped(10, &["thumb", "hero"]));
        cache.put("c".to_string(), grouped(10, &["hero"]));
        cache.put("d".to_string(), img(10));
        assert_eq!(cache.purge_group("thumb"), 2);
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), None);
        assert!(cache.get("c").is_some());
        assert!(cache.get("d").is_some());
        assert_eq!(cache.purge_group("thumb"), 0);
        assert_eq!(cache.purge_group("missing"), 0);
    }

    #[test]
    fn merge_groups_accumulates_provenance() {
        let cache = Cache::new(ROOMY);
        cache.put("a".to_string(), grouped(10, &["thumb"]));
        // A later spelling of the same entry brings its
        // own group: the stored set is the union, so a
        // group purge finds the entry either way (R2-15).
        assert!(cache.merge_groups("a", &["hero".to_string()]));
        assert!(!cache.merge_groups("a", &["thumb".to_string()]));
        assert!(!cache.merge_groups("missing", &["hero".to_string()]));
        assert_eq!(cache.get("a").unwrap().groups, ["hero", "thumb"]);
        assert_eq!(cache.purge_group("hero"), 1);
    }

    #[test]
    fn purge_source_removes_every_derivative_of_the_source() {
        let cache = Cache::new(ROOMY);
        cache.put("300x200/file:///heroes/cat.jpg".to_string(), img(10));
        cache.put(
            "300x200/file:///heroes/cat.jpg#cfgab12#avif".to_string(),
            img(10),
        );
        cache.put("600x400/file:///heroes/cat.jpg".to_string(), img(10));
        cache.put("300x200/file:///heroes/cat.jpg2".to_string(), img(10));
        cache.put("300x200/file:///heroes/dog.jpg".to_string(), img(10));
        assert_eq!(cache.purge_source("file:///heroes/cat.jpg"), 3);
        assert!(cache.get("300x200/file:///heroes/cat.jpg").is_none());
        assert!(cache.get("300x200/file:///heroes/cat.jpg2").is_some());
        assert!(cache.get("300x200/file:///heroes/dog.jpg").is_some());
        assert_eq!(cache.purge_source("file:///heroes/cat.jpg"), 0);
    }

    #[test]
    fn replacing_a_key_does_not_double_count() {
        let cache = Cache::new(cost("other", 50));
        cache.put("k".to_string(), img(50));
        cache.put("k".to_string(), img(50));
        assert_eq!(cache.get("k"), Some(img(50)));
        // Room for exactly the one entry: a second key evicts it.
        cache.put("other".to_string(), img(50));
        assert_eq!(cache.get("k"), None);
        assert_eq!(cache.get("other"), Some(img(50)));
    }

    #[test]
    fn remove_drops_exactly_one_entry() {
        let cache = Cache::new(ROOMY);
        cache.put("a".to_string(), img(10));
        cache.put("a#avif".to_string(), img(10));
        assert!(cache.remove("a"));
        assert_eq!(cache.get("a"), None);
        // The variant is a different key: untouched.
        assert!(cache.get("a#avif").is_some());
        assert!(!cache.remove("a"));
        assert!(!cache.remove("missing"));
    }

    #[test]
    fn long_keys_with_tiny_bodies_are_evicted_by_their_full_charge() {
        // A 70-byte body under a 4 KiB key keeps the key
        // resident twice. Charging only the body let a 64 KiB
        // budget hold every one of these entries, about
        // twelve times the budget in keys, and never evict.
        let budget = 64 * 1024;
        let cache = Cache::new(budget);
        let keys: Vec<String> = (0..100)
            .map(|i| format!("{i:04}/{}", "k".repeat(4091)))
            .collect();
        for key in &keys {
            cache.put(key.clone(), img(70));
        }
        assert!(cache.charged_bytes() <= budget);
        let kept: Vec<&String> = keys.iter().filter(|k| cache.get(k).is_some()).collect();
        let resident: usize = kept.iter().map(|k| 2 * k.len() + 70).sum();
        assert!(
            resident <= budget,
            "{} entries hold {resident} bytes of keys and bodies under a {budget}-byte budget",
            kept.len()
        );
        assert!(
            cache.get(&keys[0]).is_none(),
            "the oldest entry was evicted"
        );
        assert!(cache.get(&keys[99]).is_some(), "the newest entry is kept");
    }

    #[test]
    fn an_entry_whose_charge_exceeds_the_budget_is_not_stored() {
        // The body alone fits; with its key it does not.
        let key = "k".repeat(4096);
        let cache = Cache::new(4096);
        cache.put(key.clone(), img(10));
        assert_eq!(cache.get(&key), None);
        assert_eq!(cache.charged_bytes(), 0);
    }

    #[test]
    fn a_merge_that_grows_an_entry_is_charged_and_keeps_the_budget() {
        let cache = Cache::new(cost("a", 10) + cost("b", 10));
        cache.put("a".to_string(), img(10));
        cache.put("b".to_string(), img(10));
        let group = "g".repeat(64);
        assert!(cache.merge_groups("b", std::slice::from_ref(&group)));
        assert_eq!(
            cache.charged_bytes(),
            entry_cost("b", &grouped(10, &[group.as_str()])),
            "b is charged for its new group; a made room"
        );
        assert_eq!(
            cache.get("a"),
            None,
            "the least recently used entry made room"
        );
        assert_eq!(cache.get("b").unwrap().groups, [group]);
    }

    #[test]
    fn every_removal_takes_back_what_the_store_charged() {
        let cache = Cache::new(ROOMY);
        let source = "file:///heroes/cat.jpg";
        let keys = [
            format!("300x200/{source}"),
            format!("300x200/{source}#cfg1#avif"),
            format!("600x400/{source}"),
            "a/b.png".to_string(),
            "c/d.png".to_string(),
        ];
        for key in &keys {
            cache.put(key.clone(), grouped(10, &["thumb"]));
        }
        let charged: usize = keys
            .iter()
            .map(|k| entry_cost(k, &grouped(10, &["thumb"])))
            .sum();
        assert_eq!(cache.charged_bytes(), charged);
        // A merge grows two entries' charges before they go.
        assert!(cache.merge_groups(&keys[0], &["hero".to_string()]));
        assert!(cache.merge_groups(&keys[3], &["hero".to_string()]));
        assert_eq!(cache.purge_source(source), 3);
        assert_eq!(cache.purge_matching("a/b.png"), 1);
        assert!(cache.remove("c/d.png"));
        assert_eq!(cache.charged_bytes(), 0);
        cache.put("e".to_string(), grouped(10, &["thumb"]));
        assert_eq!(cache.purge_group("thumb"), 1);
        assert_eq!(cache.charged_bytes(), 0);
    }

    #[test]
    fn key_bases_within_the_bound_are_kept_as_written() {
        for base in [
            "300x200/http://x/cat.jpg".to_string(),
            "x".repeat(MAX_KEY_BASE_BYTES),
        ] {
            assert_eq!(bounded_key_base(base.clone(), "http://x/cat.jpg"), base);
        }
    }

    #[test]
    fn an_over_long_key_base_is_carried_as_a_digest_with_its_source() {
        let source = "file:///heroes/cat.jpg";
        let long_ops = format!("filters:{}", ["blur(1)"; 800].join(":"));
        let base = format!("{long_ops}/{source}");
        assert!(base.len() > MAX_KEY_BASE_BYTES);
        let bounded = bounded_key_base(base.clone(), source);
        assert!(bounded.len() <= MAX_KEY_BASE_BYTES, "{}", bounded.len());
        assert!(bounded.starts_with(DIGEST_BASE_MARK));
        // The digest covers the whole base: another chain on
        // the same source is a different key.
        let other = bounded_key_base(format!("{long_ops}:blur(2)/{source}"), source);
        assert_ne!(bounded, other);
        assert_eq!(bounded_key_base(base, source), bounded, "deterministic");
        // The per-source purge and the per-URL purge still
        // find the entry and its variants.
        let key = format!("{bounded}#cfgab12#avif");
        assert!(key_belongs_to_source(&key, source));
        assert!(!key_belongs_to_source(&key, "file:///heroes/dog.jpg"));
        assert!(key_has_base(&key, &bounded));
        assert!(!key_has_base(&key, &other));
    }

    #[test]
    fn an_over_long_source_is_carried_in_the_digest_alone() {
        // A source padded with empty path segments, the shape
        // that grew keys to tens of kilobytes.
        let source = format!("http://origin{}t.png", "/".repeat(58_000));
        let base = format!("filters:format(png)/{source}");
        let bounded = bounded_key_base(base, &source);
        assert_eq!(bounded.len(), DIGEST_BASE_MARK.len() + 64);
        // A request target holds no space, so no base a
        // request spells out collides with a digest form.
        assert!(bounded.contains(' '));
        // Without a source tail (a purge target that does not
        // parse), the digest stands alone too.
        let unparsed = "u".repeat(MAX_KEY_BASE_BYTES + 1);
        assert_eq!(
            bounded_key_base(unparsed, "").len(),
            DIGEST_BASE_MARK.len() + 64
        );
    }

    #[test]
    fn purge_matchers_agree_with_the_spelled_out_rules() {
        // Issue #462: the matchers run on every disk and
        // shared hit while a tombstone lives, and now match
        // in place instead of building a `{base}#` prefix or
        // a `/{source}` suffix on each call. They must accept
        // exactly what the spelled-out rules accept.
        let keys = [
            "",
            "#",
            "x.jpg",
            "x.jpg#w2",
            "/x.jpg",
            "a/x.jpg",
            "a/x.jpg#avif",
            "a/x.jpg#cfg1#webp",
            "a/x.jpg2",
            "ba/x.jpg",
            "a/b/x.jpg",
            "a/x.jpg/x.jpg",
        ];
        let names = ["", "x.jpg", "a/x.jpg", "b/x.jpg", "a/x", "/x.jpg"];
        for key in keys {
            let base = key.split('#').next().unwrap();
            for name in names {
                assert_eq!(
                    key_has_base(key, name),
                    key == name || key.starts_with(&format!("{name}#")),
                    "{key:?} under {name:?}"
                );
                assert_eq!(
                    key_belongs_to_source(key, name),
                    base == name || base.ends_with(&format!("/{name}")),
                    "{key:?} from {name:?}"
                );
            }
        }
    }

    #[test]
    fn purge_guard_covers_only_writes_that_predate_the_purge() {
        let guard = PurgeGuard::new();
        let before = guard.snapshot();
        guard.record_key("30x20/http://x/cat.jpg");
        // A write begun before the purge is covered, under
        // the bare key and under a variant spelling.
        assert!(guard.purge_covers(before, "30x20/http://x/cat.jpg", &[]));
        assert!(guard.purge_covers(before, "30x20/http://x/cat.jpg#avif", &[]));
        // A prefix that is not a variant boundary is a
        // different key.
        assert!(!guard.purge_covers(before, "30x20/http://x/cat.jpg2", &[]));
        // An unrelated key is not covered.
        assert!(!guard.purge_covers(before, "30x20/http://x/dog.jpg", &[]));
        // A write begun after the purge is not covered.
        let after = guard.snapshot();
        assert!(!guard.purge_covers(after, "30x20/http://x/cat.jpg", &[]));
    }

    #[test]
    fn purge_guard_matches_groups_and_sources() {
        let guard = PurgeGuard::new();
        let before = guard.snapshot();
        guard.record_group("hero");
        guard.record_source("file:///heroes/cat.jpg");
        assert!(guard.purge_covers(before, "any/key", &["hero".to_string()]));
        assert!(!guard.purge_covers(before, "any/key", &["thumb".to_string()]));
        assert!(!guard.purge_covers(before, "any/key", &[]));
        // Source purges cover every derivative key of the
        // source, and nothing else.
        assert!(guard.purge_covers(before, "300x200/file:///heroes/cat.jpg", &[]));
        assert!(guard.purge_covers(before, "300x200/file:///heroes/cat.jpg#cfg1", &[]));
        assert!(!guard.purge_covers(before, "300x200/file:///heroes/cat.jpg2", &[]));
        // Later purges of other targets do not retroactively
        // cover a snapshot taken after them.
        let mid = guard.snapshot();
        guard.record_group("thumb");
        assert!(!guard.purge_covers(mid, "any/key", &["hero".to_string()]));
        assert!(guard.purge_covers(mid, "any/key", &["thumb".to_string()]));
    }

    /// A stamp `offset_ms` from `cutoff`'s start, by the
    /// purging process or by another one.
    fn stamp_near(cutoff: &PurgeCutoff, offset_ms: i64, same_writer: bool) -> Option<WriteStamp> {
        Some(WriteStamp {
            at_ms: cutoff.start_ms.saturating_add_signed(offset_ms),
            writer: if same_writer {
                cutoff.writer
            } else {
                cutoff.writer ^ 1
            },
        })
    }

    #[test]
    fn a_cutoff_covers_copies_written_before_the_purge_began() {
        // Issues #446, #461: a copy rendered before the
        // purge began is purged, one rendered after it is
        // not, and an unstamped copy predates every purge.
        let cutoff = PurgeCutoff {
            start_ms: 1_000_000,
            writer: 42,
        };
        assert!(cutoff.covers(None));
        // The purging process shares the purge's clock:
        // the line is exact.
        assert!(cutoff.covers(stamp_near(&cutoff, -1, true)));
        assert!(!cutoff.covers(stamp_near(&cutoff, 0, true)));
        assert!(!cutoff.covers(stamp_near(&cutoff, 1, true)));
        // Another process's clock may run ahead: its copies
        // count as pre-purge until the margin has passed.
        assert!(cutoff.covers(stamp_near(&cutoff, 0, false)));
        let margin = CLOCK_SKEW_MARGIN_MS as i64;
        assert!(cutoff.covers(stamp_near(&cutoff, margin - 1, false)));
        assert!(!cutoff.covers(stamp_near(&cutoff, margin, false)));
        // The unbounded purge forms cover everything.
        assert!(PurgeCutoff::EVERYTHING.covers(stamp_near(&cutoff, margin * 1000, false)));
        assert!(PurgeCutoff::EVERYTHING.covers(None));
    }

    #[test]
    fn a_running_purge_covers_only_copies_written_before_it_began() {
        // Issue #446: while a purge runs, a persisted copy
        // it covers is a miss only if it predates the
        // purge; a render that starts after `begin` keeps
        // its copies.
        let guard = Arc::new(PurgeGuard::new());
        let key_purge = guard.begin(PurgeTarget::Key("30x20/http://x/cat.jpg".into()));
        let cutoff = key_purge.tombstone().cutoff;
        let before = stamp_near(&cutoff, -1, true).unwrap();
        let after = guard.stamp();
        let key = "30x20/http://x/cat.jpg#w300";
        assert!(guard.running_covers(key, &[], None));
        assert!(guard.running_covers("30x20/http://x/cat.jpg", &[], Some(before)));
        assert!(guard.running_covers(key, &[], stamp_near(&cutoff, -1, true)));
        assert!(!guard.running_covers(key, &[], Some(after)));
        assert!(!guard.running_covers("30x20/http://x/cat.jpg2", &[], None));
        assert!(!guard.running_covers("30x20/http://x/dog.jpg", &[], None));
        // A copy another replica wrote just after the purge
        // began is inside the skew margin; one well after
        // it is not.
        assert!(guard.running_covers(key, &[], stamp_near(&cutoff, 1, false)));
        let late = CLOCK_SKEW_MARGIN_MS as i64 + 1;
        assert!(!guard.running_covers(key, &[], stamp_near(&cutoff, late, false)));
        // Writes: one begun before `begin` is stale by its
        // generation; a pre-purge copy written back later
        // is stale by its stamp; a post-purge render is not.
        assert!(guard.purge_covers(0, "30x20/http://x/cat.jpg", &[]));
        let now = guard.snapshot();
        assert!(guard.write_covered(now, key, &[], Some(before)));
        assert!(!guard.write_covered(now, key, &[], Some(after)));
        let group_purge = guard.begin(PurgeTarget::Group("hero".into()));
        let source_purge = guard.begin(PurgeTarget::Source("file:///heroes/dog.jpg".into()));
        assert_eq!(guard.active_count(), 3);
        let hero = ["hero".to_string()];
        assert!(guard.running_covers("any/key", &hero, Some(before)));
        assert!(!guard.running_covers("any/key", &["thumb".to_string()], None));
        assert!(guard.running_covers("300x200/file:///heroes/dog.jpg#cfg1", &[], None));
        assert!(!guard.running_covers("300x200/file:///heroes/dog.jpg2", &[], None));
        drop(key_purge);
        drop(group_purge);
        drop(source_purge);
        assert_eq!(guard.active_count(), 0);
    }

    #[test]
    fn an_ended_purge_covers_older_copies_for_its_grace_only() {
        // Issues #400, #461: a lookup that read a copy
        // before the walk deleted it may check it after the
        // walk ended, and a pre-purge copy can still be
        // written back by a render that was already running
        // somewhere. An ended purge keeps covering copies
        // older than itself for its grace period, then
        // stops.
        let guard = Arc::new(PurgeGuard::new());
        let purge = guard.begin(PurgeTarget::Group("hero".into()));
        let before = stamp_near(&purge.tombstone().cutoff, -1, true);
        let after = Some(guard.stamp());
        drop(purge);
        let hero = ["hero".to_string()];
        assert!(guard.running_covers("any/key", &hero, before));
        assert!(guard.running_covers("any/key", &hero, None));
        assert!(!guard.running_covers("any/key", &hero, after));
        assert!(!guard.running_covers("any/key", &["thumb".to_string()], None));
        let past_grace = now_ms() + TOMBSTONE_GRACE_MS + 1;
        assert!(!guard.running_covers_at(past_grace, "any/key", &hero, None));
        // Ending a purge records no generation tombstone: a
        // write begun during the purge is not stale.
        let during = guard.snapshot();
        assert!(!guard.purge_covers(during, "any/key", &hero));
        // Only the most recent endings are kept.
        for i in 0..MAX_TOMBSTONES {
            drop(guard.begin(PurgeTarget::Key(format!("k{i}"))));
        }
        assert!(!guard.running_covers("any/key", &hero, None));
        assert!(guard.running_covers("k0#w300", &[], None));
    }

    #[test]
    fn shared_tombstones_cover_until_their_grace_passes() {
        // Issue #461: tombstones read from the shared
        // bucket cover this process's copies by the same
        // rule as its own purges.
        let guard = PurgeGuard::new();
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 7,
        };
        let running = Tombstone {
            target: PurgeTarget::Group("hero".into()),
            cutoff,
            ended_ms: None,
        };
        let ended = Tombstone {
            target: PurgeTarget::Source("file:///a.jpg".into()),
            cutoff,
            ended_ms: Some(cutoff.start_ms),
        };
        guard.replace_shared(HashMap::from([
            ("purges/a".to_string(), running),
            ("purges/b".to_string(), ended),
        ]));
        let hero = ["hero".to_string()];
        let own = guard.stamp();
        assert!(guard.running_covers("k", &hero, None));
        assert!(guard.running_covers("k", &hero, stamp_near(&cutoff, -10, false)));
        // This process's copy written now is inside the
        // margin of another process's purge that began now.
        assert!(guard.running_covers("k", &hero, Some(own)));
        assert!(!guard.running_covers("k", &hero, stamp_near(&cutoff, 6_000, false)));
        assert!(guard.running_covers("1x1/file:///a.jpg", &[], None));
        let later = cutoff.start_ms + TOMBSTONE_GRACE_MS + 1;
        assert!(!guard.running_covers_at(later, "1x1/file:///a.jpg", &[], None));
        // A tombstone whose walk never ended never expires.
        assert!(guard.running_covers_at(later, "k", &hero, None));
        // A refresh replaces the whole set.
        guard.replace_shared(HashMap::new());
        assert!(!guard.running_covers("k", &hero, None));
    }

    #[test]
    fn an_ended_purge_supersedes_older_tombstones_of_its_target() {
        let target = PurgeTarget::Group("hero".into());
        let tombstone = |start_ms, writer, ended_ms| Tombstone {
            target: target.clone(),
            cutoff: PurgeCutoff { start_ms, writer },
            ended_ms,
        };
        let crashed = tombstone(1_000, 1, None);
        // A later walk by the same process that ended.
        assert!(tombstone(2_000, 1, Some(9_000)).supersedes(&crashed));
        // A walk still running supersedes nothing.
        assert!(!tombstone(2_000, 1, None).supersedes(&crashed));
        // Another process: only when it began at least the
        // margin later, since each covers the other's copies
        // with the margin.
        assert!(!tombstone(2_000, 2, Some(9_000)).supersedes(&crashed));
        let far = 1_000 + CLOCK_SKEW_MARGIN_MS;
        assert!(tombstone(far, 2, Some(far)).supersedes(&crashed));
        // Never an equal cutoff (two copies of one purge
        // must not remove each other), an older one, or
        // another target.
        assert!(!tombstone(1_000, 1, Some(9_000)).supersedes(&crashed));
        assert!(!crashed.supersedes(&tombstone(500, 1, Some(9_000))));
        let other = Tombstone {
            target: PurgeTarget::Group("thumb".into()),
            ..tombstone(2_000, 1, Some(9_000))
        };
        assert!(!other.supersedes(&crashed));
    }

    #[test]
    fn active_purge_ends_when_its_task_panics() {
        let guard = Arc::new(PurgeGuard::new());
        let task_guard = guard.clone();
        let joined = std::thread::spawn(move || {
            let _active = task_guard.begin(PurgeTarget::Group("hero".into()));
            panic!("purge task panics mid-walk");
        })
        .join();
        assert!(joined.is_err());
        assert_eq!(guard.active_count(), 0);
        // The tombstone outlives the task for its grace,
        // ended, not running.
        let fresh = guard.stamp();
        assert!(!guard.running_covers("any/key", &["hero".to_string()], Some(fresh)));
    }

    #[test]
    fn writer_ids_differ_per_guard() {
        // Each boot gets its own writer id, so a restarted
        // process compares its new copies against its old
        // purges with the margin.
        assert_ne!(PurgeGuard::new().writer, PurgeGuard::new().writer);
    }

    #[test]
    fn one_process_orders_its_stamps_and_purges_by_event_not_by_wall_clock() {
        // A copy and a purge of one process compare with no
        // margin, so their times must follow the order of
        // events. A copy rendered before a purge in the same
        // millisecond, or before the host clock stepped
        // back, must still be covered, or the walk would
        // leave it in place for good; one rendered after the
        // purge must not be, or it would churn.
        let guard = PurgeGuard::new();
        let own = |at_ms| {
            Some(WriteStamp {
                at_ms,
                writer: guard.writer,
            })
        };
        let cutoff = |start_ms| PurgeCutoff {
            start_ms,
            writer: guard.writer,
        };
        // The same millisecond: the purge starts after the
        // render that preceded it.
        let render = guard.stamp_ms(10_000);
        let start = guard.purge_start_ms(10_000);
        assert!(cutoff(start).covers(own(render)));
        let after = guard.stamp_ms(10_000);
        assert!(!cutoff(start).covers(own(after)));
        // The clock steps back 3 s between a render and a
        // purge: the purge still starts after the render.
        let render = guard.stamp_ms(20_000);
        let start = guard.purge_start_ms(17_000);
        assert!(start > render);
        assert!(cutoff(start).covers(own(render)));
        // A render after that purge, before the wall clock
        // has caught up, is not covered by it.
        let after = guard.stamp_ms(17_001);
        assert!(after >= start);
        assert!(!cutoff(start).covers(own(after)));
        // Once the wall clock passes, times follow it again.
        assert_eq!(guard.stamp_ms(30_000), 30_000);
        assert_eq!(guard.purge_start_ms(40_000), 40_000);
        // The public paths use the same clock.
        let purge = Arc::new(PurgeGuard::new());
        let before = purge.stamp();
        let active = purge.begin(PurgeTarget::Group("hero".into()));
        assert!(active.tombstone().cutoff.start_ms > before.at_ms);
        assert!(purge.stamp().at_ms >= active.tombstone().cutoff.start_ms);
    }

    #[test]
    fn purge_guard_keeps_only_recent_tombstones() {
        let guard = PurgeGuard::new();
        let start = guard.snapshot();
        for i in 0..(MAX_TOMBSTONES + 5) {
            guard.record_key(&format!("key-{i}"));
        }
        // The newest tombstones are retained...
        assert!(guard.purge_covers(start, &format!("key-{}", MAX_TOMBSTONES + 4), &[]));
        // ...and the oldest have been trimmed away.
        assert!(!guard.purge_covers(start, "key-0", &[]));
    }
}
