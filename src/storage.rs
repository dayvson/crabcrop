//! Disk-backed result storage: the persistent tier under the
//! in-memory result cache.
//!
//! Entries live one per file under a configured directory,
//! named by the SHA-256 of their cache key (keys contain
//! slashes and cannot be paths). A file is a small text header
//! (format, quality, and the original key, so purge-by-path
//! works after a restart) followed by the raw image bytes.
//! Writes go to a temporary file that is renamed into place,
//! so a crash can leave an orphan temp file but never a
//! partial entry; a file that fails to parse is a miss, is
//! deleted, and is overwritten by the next store. The byte
//! budget evicts least recently used entries, with recency
//! tracked in memory and re-seeded from file times at open.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::cache::CachedImage;
use crate::url::ImageFormat;

#[derive(Debug, Default)]
struct Inner {
    /// Cache key -> (file stem, total file size, cache groups).
    entries: HashMap<String, (String, usize, Vec<String>)>,
    /// Keys from least to most recently used.
    order: VecDeque<String>,
    total_bytes: usize,
}

/// A byte-bounded, LRU-evicting file store. A `max_bytes` of 0
/// disables storage. All operations are best-effort: IO errors
/// degrade to misses and dropped writes, never failures,
/// because a cache must not take the pipeline down with it.
#[derive(Debug)]
pub struct DiskStore {
    dir: PathBuf,
    max_bytes: usize,
    inner: Mutex<Inner>,
}

/// Entry format version. CRABDISK2 adds the entry's cache
/// groups to the header; older files do not parse and are
/// deleted on sight at scan, the store's standing upgrade
/// behavior for a cache.
const MAGIC: &str = "CRABDISK2";

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

impl DiskStore {
    /// Open (creating if needed) the store rooted at `dir`,
    /// indexing whatever entries a previous run left behind.
    pub fn open(dir: &Path, max_bytes: usize) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create the result storage directory: {e}"))?;
        let store = Self {
            dir: dir.to_path_buf(),
            max_bytes,
            inner: Mutex::new(Inner::default()),
        };
        store.scan()?;
        Ok(store)
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
            match read_header(&path) {
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
            inner.total_bytes += size;
            inner.order.push_back(key.clone());
            inner.entries.insert(key, (stem, size, groups));
        }
        Ok(())
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
        let mut inner = self.inner.lock().unwrap();
        let (stem, _, _) = inner.entries.get(key)?.clone();
        let path = self.file_for(&stem);
        let parsed = std::fs::read(&path)
            .ok()
            .and_then(|data| parse_entry(&data));
        match parsed {
            Some((stored_key, image)) if stored_key == key => {
                touch(&mut inner, key);
                Some(image)
            }
            _ => {
                evict_entry(&mut inner, key);
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    /// Store `image` under `key`, evicting least recently used
    /// entries until the new file fits the budget.
    pub fn put(&self, key: &str, image: &CachedImage) {
        if self.max_bytes == 0 {
            return;
        }
        let data = serialize_entry(key, image);
        if data.len() > self.max_bytes {
            return;
        }
        let stem = key_stem(key);
        let tmp = self.dir.join(tmp_name(&stem));
        if std::fs::write(&tmp, &data).is_err() {
            return;
        }
        if std::fs::rename(&tmp, self.file_for(&stem)).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        if let Some((_, old_size, _)) = inner.entries.remove(key) {
            inner.total_bytes -= old_size;
            remove_from_order(&mut inner, key);
        }
        while inner.total_bytes + data.len() > self.max_bytes {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            if let Some((old_stem, size, _)) = inner.entries.remove(&oldest) {
                inner.total_bytes -= size;
                let _ = std::fs::remove_file(self.file_for(&old_stem));
            }
        }
        inner.total_bytes += data.len();
        inner.order.push_back(key.to_string());
        inner
            .entries
            .insert(key.to_string(), (stem, data.len(), image.groups.clone()));
    }

    /// Remove every entry for `base`: the bare key plus its
    /// `#`-suffixed variants (Accept class, DPR). Returns how
    /// many entries were removed.
    pub fn purge_matching(&self, base: &str) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let keys: Vec<String> = inner
            .entries
            .keys()
            .filter(|k| k.as_str() == base || k.starts_with(&format!("{base}#")))
            .cloned()
            .collect();
        let mut removed = 0;
        for key in keys {
            if let Some((stem, _)) = evict_entry(&mut inner, &key) {
                let _ = std::fs::remove_file(self.file_for(&stem));
                removed += 1;
            }
        }
        removed
    }

    /// Remove every entry belonging to `group` (RFC 9875
    /// grouped invalidation), using the groups indexed at
    /// scan and at store time. Returns how many entries
    /// were removed.
    pub fn purge_group(&self, group: &str) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let keys: Vec<String> = inner
            .entries
            .iter()
            .filter(|(_, (_, _, groups))| groups.iter().any(|g| g == group))
            .map(|(key, _)| key.clone())
            .collect();
        let mut removed = 0;
        for key in keys {
            if let Some((stem, _)) = evict_entry(&mut inner, &key) {
                let _ = std::fs::remove_file(self.file_for(&stem));
                removed += 1;
            }
        }
        removed
    }

    /// (entries, total bytes) currently indexed, for the boot
    /// log line.
    pub fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.entries.len(), inner.total_bytes)
    }
}

fn key_stem(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
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

fn evict_entry(inner: &mut Inner, key: &str) -> Option<(String, usize)> {
    let (stem, size, _) = inner.entries.remove(key)?;
    inner.total_bytes -= size;
    remove_from_order(inner, key);
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
        _ => return None,
    })
}

fn serialize_entry(key: &str, image: &CachedImage) -> Vec<u8> {
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
    let head = format!(
        "{MAGIC}\n{}\n{quality}\n{}\n{}\n{}\n",
        format_name(image.format),
        image.groups.len(),
        key.len(),
        image.bytes.len()
    );
    let mut out =
        Vec::with_capacity(head.len() + key.len() + 1 + groups_block.len() + image.bytes.len());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(key.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(&groups_block);
    out.extend_from_slice(&image.bytes);
    out
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

/// Split a stored file into (key, format, quality, groups,
/// payload offset). `None` on any inconsistency, including a
/// payload whose length disagrees with the header: the header
/// is the integrity check.
/// A stored file's parsed header.
struct EntryHeader {
    key: String,
    format: ImageFormat,
    quality: Option<u8>,
    groups: Vec<String>,
    payload_offset: usize,
}

fn parse_header_bytes(data: &[u8]) -> Option<EntryHeader> {
    let mut pos = 0;
    if take_line(data, &mut pos)? != MAGIC.as_bytes() {
        return None;
    }
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
    let key_bytes = data.get(pos..pos + key_len)?;
    if data.get(pos + key_len) != Some(&b'\n') {
        return None;
    }
    let key = String::from_utf8(key_bytes.to_vec()).ok()?;
    let mut cursor = pos + key_len + 1;
    let mut groups = Vec::with_capacity(group_count);
    for _ in 0..group_count {
        let len_line = std::str::from_utf8(take_line(data, &mut cursor)?).ok()?;
        let glen: usize = len_line.parse().ok()?;
        let gbytes = data.get(cursor..cursor + glen)?;
        if data.get(cursor + glen) != Some(&b'\n') {
            return None;
        }
        groups.push(String::from_utf8(gbytes.to_vec()).ok()?);
        cursor += glen + 1;
    }
    let payload_offset = cursor;
    if data.len() != payload_offset + payload_len {
        return None;
    }
    Some(EntryHeader {
        key,
        format,
        quality,
        groups,
        payload_offset,
    })
}

fn parse_entry(data: &[u8]) -> Option<(String, CachedImage)> {
    let header = parse_header_bytes(data)?;
    let bytes = data.get(header.payload_offset..)?.to_vec();
    if bytes.is_empty() {
        return None;
    }
    Some((
        header.key,
        CachedImage {
            bytes,
            format: header.format,
            quality: header.quality,
            groups: header.groups,
        },
    ))
}

/// Just the header of a stored file, for the boot scan.
fn read_header(path: &Path) -> Option<(String, Vec<String>)> {
    let data = std::fs::read(path).ok()?;
    let header = parse_header_bytes(&data)?;
    Some((header.key, header.groups))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_names_are_unique_per_put() {
        // Concurrent puts of one key must never share a
        // temp file (review P-6).
        assert_ne!(tmp_name("abc"), tmp_name("abc"));
        assert!(tmp_name("abc").starts_with("abc.tmp-"));
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("crabcrop-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn image(bytes: &[u8]) -> CachedImage {
        CachedImage {
            bytes: bytes.to_vec(),
            format: ImageFormat::Png,
            quality: Some(80),
            groups: Vec::new(),
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
            let store = DiskStore::open(&dir, 1_000_000).unwrap();
            store.put("300x200/http://x/a.png", &img);
            assert_eq!(store.get("300x200/http://x/a.png"), Some(img.clone()));
        }
        // A fresh instance over the same directory still has it,
        // format and quality intact.
        let store = DiskStore::open(&dir, 1_000_000).unwrap();
        assert_eq!(store.get("300x200/http://x/a.png"), Some(img));
        assert_eq!(store.stats().0, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_respects_the_byte_budget() {
        let dir = temp_dir("evict");
        let store = DiskStore::open(&dir, 300).unwrap();
        let payload = vec![7u8; 100];
        store.put("a", &image(&payload));
        store.put("b", &image(&payload));
        store.put("c", &image(&payload));
        // Each entry is ~125 bytes on disk; 300 fits two, so
        // the third store evicted the least recently used.
        assert!(store.get("c").is_some());
        let (_, total) = store.stats();
        assert!(total <= 300, "total {total}");
        assert!(store.get("a").is_none(), "the LRU entry went first");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_file_is_a_miss_and_is_removed() {
        let dir = temp_dir("corrupt");
        let store = DiskStore::open(&dir, 1_000_000).unwrap();
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
            let store = DiskStore::open(&dir, 1_000_000).unwrap();
            store.put("a", &grouped_image(b"1", &["thumb"]));
            store.put("b", &grouped_image(b"2", &["thumb", "hero"]));
            store.put("c", &grouped_image(b"3", &["hero"]));
            assert_eq!(store.get("b").unwrap().groups, ["thumb", "hero"]);
        }
        // Reopen: the scan re-indexes groups from the file
        // headers, so group purge works across restarts.
        let store = DiskStore::open(&dir, 1_000_000).unwrap();
        assert_eq!(store.purge_group("thumb"), 2);
        assert!(store.get("a").is_none());
        assert!(store.get("b").is_none());
        assert!(store.get("c").is_some());
        assert_eq!(store.purge_group("thumb"), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_removes_the_base_and_its_variants_only() {
        let dir = temp_dir("purge");
        let store = DiskStore::open(&dir, 1_000_000).unwrap();
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
}
