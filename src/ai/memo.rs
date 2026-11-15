//! Recent detections, remembered by the digest of the
//! model's input (issue #677).
//!
//! The engine shares one detection between the operations of
//! a request, but nothing was shared between requests: the
//! same source rendered at another size ran the detector
//! again on the same pixels. A smart crop detects on the
//! source before it cuts, so `crop:400x300:smart` and
//! `crop:800x600:smart` of one photo feed the face model the
//! same input, and each is its own miss in the result cache.
//!
//! A model is a function of its input, so a provider keeps
//! the boxes it last answered, keyed by the SHA-256 of
//! everything that decides them: the pixels the model's
//! input is filled from and whatever the boxes are mapped
//! back with. A hit hands back the boxes a fresh run would
//! have. The memo holds keys and boxes, never pixels; it is
//! bounded in entries and in boxes per entry; and it never
//! stands between a caller and the model: a lookup it cannot
//! make is a miss, and a result it cannot keep is not kept.
//! It is a field of its provider and goes when the provider
//! goes.

use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::Mutex;

/// How many results a provider remembers.
const MEMO_ENTRIES: usize = 256;

/// The most boxes a remembered result holds. A result with
/// more (a crowd, or a detector declared with a score floor
/// near zero) is not remembered, so the bound on entries is
/// a bound on memory too: a full memo of faces stays under
/// 350 KB, and one of typical results is a few tens of
/// kilobytes.
const MEMO_BOXES: usize = 64;

/// What a result is remembered under: the SHA-256 of
/// everything that decides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemoKey([u8; 32]);

impl MemoKey {
    /// The digest of the parts in order, each behind its
    /// length, so that no two lists of parts hash the same
    /// bytes.
    pub(super) fn of(parts: &[&[u8]]) -> Self {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update((part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&hasher.finalize());
        Self(key)
    }
}

/// A provider's most recent results, most recently used
/// first. Safe to share between threads: every operation
/// takes the lock for a scan of at most [`MEMO_ENTRIES`]
/// keys, and no model runs while it is held.
pub(super) struct DetectionMemo<B> {
    entries: Mutex<VecDeque<(MemoKey, Vec<B>)>>,
    capacity: usize,
}

impl<B: Clone> DetectionMemo<B> {
    pub(super) fn new() -> Self {
        Self::with_capacity(MEMO_ENTRIES)
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            capacity,
        }
    }

    /// The boxes remembered under `key`, which becomes the
    /// most recently used. `None` is a miss, and so is a
    /// lock another thread poisoned: the caller runs the
    /// model.
    pub(super) fn get(&self, key: &MemoKey) -> Option<Vec<B>> {
        let mut entries = self.entries.lock().ok()?;
        let at = entries.iter().position(|(k, _)| k == key)?;
        let entry = entries.remove(at)?;
        let boxes = entry.1.clone();
        entries.push_front(entry);
        Some(boxes)
    }

    /// Remember a result the model just gave. The least
    /// recently used entry leaves when the memo is full. A
    /// result with more than [`MEMO_BOXES`] boxes is not
    /// remembered, and nothing is behind a poisoned lock.
    pub(super) fn put(&self, key: MemoKey, boxes: &[B]) {
        if boxes.len() > MEMO_BOXES || self.capacity == 0 {
            return;
        }
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        if let Some(at) = entries.iter().position(|(k, _)| *k == key) {
            entries.remove(at);
        }
        entries.push_front((key, boxes.to_vec()));
        entries.truncate(self.capacity);
    }
}

#[cfg(test)]
mod tests;
