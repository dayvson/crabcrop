//! Server-side result cache: processed responses keyed by the
//! verified request path.
//!
//! A cache key names a request (a transformation of a source
//! locator), not the bytes it produced: if the source changes,
//! a re-render under the same key yields different bytes. Each
//! entry therefore carries the strong ETag of its own bytes,
//! computed at render time, so conditional requests validate
//! against what was actually rendered. The cache is bounded
//! by total bytes with least-recently-used eviction. Only
//! successful responses are stored; errors are never cached.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
pub(crate) fn key_belongs_to_source(key: &str, source: &str) -> bool {
    let base = key.split('#').next().unwrap_or(key);
    base == source || base.ends_with(&format!("/{source}"))
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

/// How many tombstones of one kind the guard retains.
/// Purges are operator actions, so the distinct targets
/// stay few in practice; the cap only bounds the
/// pathological, and a write old enough to outlive this
/// many later purges is far outside any render's
/// lifetime (renders are bounded by the process timeout,
/// not by operator patience).
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
/// The guard also knows which purges are still running
/// (issue #400). A shared-tier purge walk can outlive its
/// request by minutes on a large bucket, and until it
/// reaches an object that object still reads as a hit. A
/// lookup that starts after the purge snapshots the newer
/// generation, so the tombstone alone would let it serve
/// that copy and promote it into the local tiers, where
/// the walk never looks. While a purge is active, a copy
/// it covers that a lookup finds in a persistent tier is
/// therefore treated as already purged, whatever the
/// lookup's snapshot.
///
/// The guard tracks purges, not entries: it holds no
/// image data and its maps stay proportional to how many
/// distinct targets were purged, capped above, plus the
/// purges still running.
#[derive(Debug, Default)]
pub struct PurgeGuard {
    inner: Mutex<PurgeGuardInner>,
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
    /// Purges still running (issue #400), by generation:
    /// one entry per [`ActivePurge`] handle alive. Not
    /// capped: each entry belongs to a purge task that is
    /// still doing work, and it leaves when that task ends.
    active: HashMap<u64, PurgeTarget>,
}

/// What one purge removes: the matching rule the tiers
/// remove by, kept for as long as the purge runs.
#[derive(Debug)]
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
    fn covers(&self, key: &str, groups: &[String]) -> bool {
        match self {
            PurgeTarget::Key(base) => key == base || key.starts_with(&format!("{base}#")),
            PurgeTarget::Group(group) => groups.iter().any(|g| g == group),
            PurgeTarget::Source(source) => key_belongs_to_source(key, source),
        }
    }
}

/// A purge in progress (issue #400), from its tombstone
/// until the last tier it reaches is done, the shared
/// walk's background remainder included. Dropping the
/// handle ends it, so a purge task that finishes, panics,
/// or is torn down never leaves it behind.
#[must_use = "the purge counts as active only while the handle lives"]
#[derive(Debug)]
pub struct ActivePurge {
    guard: Arc<PurgeGuard>,
    generation: u64,
}

impl Drop for ActivePurge {
    fn drop(&mut self) {
        let mut inner = self
            .guard
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.active.remove(&self.generation);
    }
}

impl PurgeGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// The current purge generation: a writer snapshots
    /// this when it starts, before it reads or renders
    /// anything it might later write back.
    pub fn snapshot(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    /// Start a purge of `target`: record its tombstone at
    /// a fresh generation and mark it active, in one step,
    /// so no lookup can fall between the two. The purge
    /// stays active until the returned handle is dropped;
    /// the caller holds it until every tier it purges is
    /// done.
    pub fn begin(self: &Arc<Self>, target: PurgeTarget) -> ActivePurge {
        let mut inner = self.inner.lock().unwrap();
        let generation = record_tombstone(&mut inner, &target);
        inner.active.insert(generation, target);
        ActivePurge {
            guard: Arc::clone(self),
            generation,
        }
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

    /// Whether a purge that is still running covers an
    /// entry stored under `key` with `groups`, whatever
    /// generation the reader snapshotted (issue #400). A
    /// tier copy it covers predates the purge or raced it,
    /// and must read as a miss until the purge has reached
    /// every tier.
    pub fn active_covers(&self, key: &str, groups: &[String]) -> bool {
        let inner = self.inner.lock().unwrap();
        inner
            .active
            .values()
            .any(|target| target.covers(key, groups))
    }

    /// How many purges are still running.
    #[cfg(test)]
    pub fn active_count(&self) -> usize {
        self.inner.lock().unwrap().active.len()
    }

    /// Whether a purge recorded after `snapshot` covers
    /// an entry stored under `key` with `groups`: such a
    /// write began before the purge and must not stand.
    pub fn purge_covers(&self, snapshot: u64, key: &str, groups: &[String]) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.keys.iter().any(|(base, purge_gen)| {
            *purge_gen > snapshot && (key == base || key.starts_with(&format!("{base}#")))
        }) || inner
            .groups
            .iter()
            .any(|(group, purge_gen)| *purge_gen > snapshot && groups.iter().any(|g| g == group))
            || inner.sources.iter().any(|(source, purge_gen)| {
                *purge_gen > snapshot && key_belongs_to_source(key, source)
            })
    }
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

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<String, CachedImage>,
    /// Last-access stamp per key (monotonic clock). Eviction
    /// removes the smallest stamp, which is the least recently
    /// used entry: the same order the VecDeque kept, at O(1)
    /// per access instead of a linear scan per hit.
    stamps: HashMap<String, u64>,
    clock: u64,
    total_bytes: usize,
}

/// A byte-bounded LRU cache. A `max_bytes` of 0 disables it.
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
        let image = inner.entries.get(key)?.clone();
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
            inner.total_bytes -= old.bytes.len();
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
        inner.total_bytes -= old.bytes.len();
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
            .extract_if(|_, image| image.groups.iter().any(|g| g == group))
        {
            inner.total_bytes -= old.bytes.len();
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
    pub fn merge_groups(&self, key: &str, groups: &[String]) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(image) = inner.entries.get_mut(key) else {
            return false;
        };
        let merged = union_groups(&image.groups, groups);
        if merged == image.groups {
            return false;
        }
        image.groups = merged;
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
            inner.total_bytes -= old.bytes.len();
            inner.stamps.remove(&key);
            removed += 1;
        }
        removed
    }

    /// Store `image` under `key`, evicting least recently used
    /// entries until it fits. Entries larger than the whole budget
    /// are not stored.
    pub fn put(&self, key: String, image: CachedImage) {
        if self.max_bytes == 0 || image.bytes.len() > self.max_bytes {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        if let Some(old) = inner.entries.remove(&key) {
            inner.total_bytes -= old.bytes.len();
            inner.stamps.remove(&key);
        }
        while inner.total_bytes + image.bytes.len() > self.max_bytes {
            let oldest = inner
                .stamps
                .iter()
                .min_by_key(|(_, stamp)| *stamp)
                .map(|(k, _)| k.clone());
            let Some(oldest) = oldest else { break };
            if let Some(evicted) = inner.entries.remove(&oldest) {
                inner.total_bytes -= evicted.bytes.len();
            }
            inner.stamps.remove(&oldest);
        }
        inner.total_bytes += image.bytes.len();
        inner.clock += 1;
        let stamp = inner.clock;
        inner.stamps.insert(key.clone(), stamp);
        inner.entries.insert(key, image);
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
        }
    }

    fn grouped(n: usize, groups: &[&str]) -> CachedImage {
        CachedImage {
            groups: groups.iter().map(|g| g.to_string()).collect(),
            ..img(n)
        }
    }

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
        let cache = Cache::new(1024);
        cache.put("a".to_string(), img(10));
        assert_eq!(cache.get("a"), Some(img(10)));
        assert_eq!(cache.get("missing"), None);
    }

    #[test]
    fn eviction_drops_the_oldest_when_over_budget() {
        let cache = Cache::new(100);
        cache.put("a".to_string(), img(60));
        cache.put("b".to_string(), img(60));
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), Some(img(60)));
    }

    #[test]
    fn get_refreshes_recency() {
        let cache = Cache::new(100);
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
        let cache = Cache::new(1024);
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
        let cache = Cache::new(1024);
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
        let cache = Cache::new(4096);
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
        let cache = Cache::new(60);
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
        let cache = Cache::new(1024);
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

    #[test]
    fn active_purges_cover_later_readers_until_they_end() {
        // Issue #400: a reader that starts after a purge
        // snapshots the newer generation, so the tombstone
        // alone lets it through. While the purge is still
        // running, the active mark does not.
        let guard = Arc::new(PurgeGuard::new());
        let key_purge = guard.begin(PurgeTarget::Key("30x20/http://x/cat.jpg".into()));
        let after = guard.snapshot();
        assert!(!guard.purge_covers(after, "30x20/http://x/cat.jpg#w300", &[]));
        assert!(guard.active_covers("30x20/http://x/cat.jpg#w300", &[]));
        assert!(guard.active_covers("30x20/http://x/cat.jpg", &[]));
        assert!(!guard.active_covers("30x20/http://x/cat.jpg2", &[]));
        assert!(!guard.active_covers("30x20/http://x/dog.jpg", &[]));
        // The mark and the tombstone go up together: a
        // write begun before `begin` is stale.
        assert!(guard.purge_covers(0, "30x20/http://x/cat.jpg", &[]));
        let group_purge = guard.begin(PurgeTarget::Group("hero".into()));
        let source_purge = guard.begin(PurgeTarget::Source("file:///heroes/dog.jpg".into()));
        assert_eq!(guard.active_count(), 3);
        assert!(guard.active_covers("any/key", &["hero".to_string()]));
        assert!(!guard.active_covers("any/key", &["thumb".to_string()]));
        assert!(guard.active_covers("300x200/file:///heroes/dog.jpg#cfg1", &[]));
        assert!(!guard.active_covers("300x200/file:///heroes/dog.jpg2", &[]));
        // Ending one purge leaves the others active, and
        // its tombstone outlives the mark.
        drop(key_purge);
        assert!(!guard.active_covers("30x20/http://x/cat.jpg#w300", &[]));
        assert!(guard.purge_covers(0, "30x20/http://x/cat.jpg", &[]));
        assert!(guard.active_covers("any/key", &["hero".to_string()]));
        drop(group_purge);
        drop(source_purge);
        assert_eq!(guard.active_count(), 0);
        assert!(!guard.active_covers("any/key", &["hero".to_string()]));
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
        assert!(!guard.active_covers("any/key", &["hero".to_string()]));
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
