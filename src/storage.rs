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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::cache::CachedImage;
use crate::url::ImageFormat;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Default)]
struct Inner {
    /// Cache key -> (file stem, total file size, cache groups).
    entries: HashMap<String, (String, usize, Vec<String>)>,
    /// Last-access stamp per key (monotonic clock). Eviction
    /// removes the smallest stamp, the least recently used
    /// entry: the same order the VecDeque kept, at O(1) per
    /// access instead of a linear scan per hit.
    stamps: HashMap<String, u64>,
    clock: u64,
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
#[derive(Debug)]
pub struct DiskStore {
    dir: PathBuf,
    max_bytes: usize,
    /// The cache entry MAC key, derived from the server
    /// secret (issue #190): entries are written and
    /// verified under it, so an entry planted in the
    /// directory by anyone without the secret is a miss.
    entry_key: Vec<u8>,
    inner: Mutex<Inner>,
}

/// Entry format version. CRABDISK2 adds the entry's cache
/// groups to the header; CRABDISK3 marks the move to
/// canonical cache keys (Q1), a one-time invalidation of
/// entries keyed by spelling; CRABDISK4 adds an
/// HMAC-SHA256 over the entry under a key derived from
/// the server secret (issue #190, audit R2-13 part a),
/// so persistent entries are authenticated, not merely
/// self-consistent. Older files do not parse and are
/// deleted on sight at scan, the store's standing upgrade
/// behavior for a cache.
const MAGIC: &str = "CRABDISK4";

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
/// unbounded allocation. `None` on IO error or an
/// oversized file; both read as a corrupt entry upstream.
fn read_capped(path: &Path, cap: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
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
    /// `entry_key` is the cache entry MAC key (see
    /// [`crate::url::entry_key`]): files that fail
    /// verification under it are deleted on sight at scan.
    pub fn open(dir: &Path, max_bytes: usize, entry_key: Vec<u8>) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create the result storage directory: {e}"))?;
        let store = Self {
            dir: dir.to_path_buf(),
            max_bytes,
            entry_key,
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
            inner.total_bytes += size;
            inner.clock += 1;
            let stamp = inner.clock;
            inner.stamps.insert(key.clone(), stamp);
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
        let stem = {
            let inner = self.inner.lock().unwrap();
            inner.entries.get(key)?.0.clone()
        };
        let path = self.file_for(&stem);
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
    /// entries until the new file fits the budget.
    pub fn put(&self, key: &str, image: &CachedImage) {
        if self.max_bytes == 0 {
            return;
        }
        let data = serialize_entry(key, image, &self.entry_key);
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
        let mut evicted: Vec<String> = Vec::new();
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some((_, old_size, _)) = inner.entries.remove(key) {
                inner.total_bytes -= old_size;
                inner.stamps.remove(key);
            }
            while inner.total_bytes + data.len() > self.max_bytes {
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
            inner.total_bytes += data.len();
            inner.clock += 1;
            let stamp = inner.clock;
            inner.stamps.insert(key.to_string(), stamp);
            inner
                .entries
                .insert(key.to_string(), (stem, data.len(), image.groups.clone()));
        }
        for old_stem in evicted {
            let _ = std::fs::remove_file(self.file_for(&old_stem));
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

    /// (entries, total bytes) currently indexed, for the boot
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
        "{}\n{quality}\n{}\n{}\n{}\n",
        format_name(image.format),
        image.groups.len(),
        key.len(),
        image.bytes.len()
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
/// payload whose length disagrees with the header.
/// `total_len` is the entry's full length: for a whole
/// entry in memory it is `data.len()`; for a header prefix
/// read on its own (the shared tier's purge walk) it is the
/// object's real length, and `data` then holds only the
/// header bytes, which is all this function touches. The
/// MAC is parsed but not verified here: verification needs
/// the whole entry, payload included, and happens in
/// [`parse_entry`] and [`read_header`], the serving paths.
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
}

fn parse_header_bytes(data: &[u8], total_len: usize) -> Option<EntryHeader> {
    let mut pos = 0;
    if take_line(data, &mut pos)? != MAGIC.as_bytes() {
        return None;
    }
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
    Some((
        header.key,
        CachedImage {
            bytes: bytes::Bytes::copy_from_slice(payload),
            format: header.format,
            quality: header.quality,
            groups: header.groups,
        },
    ))
}

/// Just the key and groups of a stored entry, WITHOUT MAC
/// verification, for the shared tier's purge walk: matching
/// a purge target needs the entry's claimed key and
/// groups, and verification needs the whole object, which
/// is exactly the full read the walk exists to avoid. A
/// forged header can at worst steer a purge; it can never
/// be served, because every serving read goes through
/// [`parse_entry`] and verifies. `total_len` is the
/// object's real length; `data` holds at least its header.
pub(crate) fn parse_entry_header(data: &[u8], total_len: usize) -> Option<(String, Vec<String>)> {
    let header = parse_header_bytes(data, total_len)?;
    Some((header.key, header.groups))
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
    fn eviction_respects_the_byte_budget() {
        let dir = temp_dir("evict");
        let store = DiskStore::open(&dir, 300, test_entry_key()).unwrap();
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
        let mut body = format!("png\n-\n{group_count}\n{key_len}\n{payload_len}\n").into_bytes();
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
}
