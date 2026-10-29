//! Server-side result cache: processed responses keyed by the
//! verified request path.
//!
//! Signed URLs are content-addressed and responses are immutable, so
//! a repeat request can be served from memory safely. The cache is
//! bounded by total bytes with least-recently-used eviction. Only
//! successful responses are stored; errors are never cached.

use std::collections::HashMap;
use std::sync::Mutex;

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
        CachedImage {
            bytes: bytes::Bytes::from(vec![7; n]),
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
}
