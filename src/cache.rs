//! Server-side result cache: processed responses keyed by the
//! verified request path.
//!
//! Signed URLs are content-addressed and responses are immutable, so
//! a repeat request can be served from memory safely. The cache is
//! bounded by total bytes with least-recently-used eviction. Only
//! successful responses are stored; errors are never cached.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::url::ImageFormat;

/// A stored response: everything needed to answer identically,
/// including the auto-quality header value when there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedImage {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    pub quality: Option<u8>,
    /// Cache groups (RFC 9875) the response belongs to: the
    /// preset names its URL referenced, empty for preset-free
    /// requests. Group purge matches on these.
    pub groups: Vec<String>,
}

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<String, CachedImage>,
    /// Keys from least to most recently used.
    order: VecDeque<String>,
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
        let mut inner = self.inner.lock().unwrap();
        // Built once, not per key examined.
        let prefix = format!("{base}#");
        let keys: Vec<String> = inner
            .entries
            .keys()
            .filter(|k| k.as_str() == base || k.starts_with(&prefix))
            .cloned()
            .collect();
        let mut removed = 0;
        for key in keys {
            if let Some(old) = inner.entries.remove(&key) {
                inner.total_bytes -= old.bytes.len();
                remove_from_order(&mut inner, &key);
                removed += 1;
            }
        }
        removed
    }

    /// Remove every entry belonging to `group` (RFC 9875
    /// grouped invalidation). Returns how many entries were
    /// removed.
    pub fn purge_group(&self, group: &str) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let keys: Vec<String> = inner
            .entries
            .iter()
            .filter(|(_, image)| image.groups.iter().any(|g| g == group))
            .map(|(key, _)| key.clone())
            .collect();
        let mut removed = 0;
        for key in keys {
            if let Some(old) = inner.entries.remove(&key) {
                inner.total_bytes -= old.bytes.len();
                remove_from_order(&mut inner, &key);
                removed += 1;
            }
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
            remove_from_order(&mut inner, &key);
        }
        while inner.total_bytes + image.bytes.len() > self.max_bytes {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            if let Some(evicted) = inner.entries.remove(&oldest) {
                inner.total_bytes -= evicted.bytes.len();
            }
        }
        inner.total_bytes += image.bytes.len();
        inner.order.push_back(key.clone());
        inner.entries.insert(key, image);
    }
}

fn touch(inner: &mut Inner, key: &str) {
    remove_from_order(inner, key);
    inner.order.push_back(key.to_string());
}

fn remove_from_order(inner: &mut Inner, key: &str) {
    if let Some(pos) = inner.order.iter().position(|k| k == key) {
        inner.order.remove(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(n: usize) -> CachedImage {
        CachedImage {
            bytes: vec![7; n],
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
