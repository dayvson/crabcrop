//! A scratch directory per test, for every test target.
//!
//! The library's unit tests reach this through
//! `crate::test_support`. The binary's unit tests and the
//! integration tests are other crates, which cannot see the
//! library's test-only modules, so they include this file
//! by path (`#[path]`), and every target shares one helper.
//! It uses only `std` for that reason.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A directory that belongs to one test, under the system
/// temp dir (`std::env::temp_dir()`, so `TMPDIR` moves
/// it). It starts empty and is removed, with everything in
/// it, when dropped, a panicking test's included.
///
/// Tests used to name their files after the process ID
/// alone (issue #526), and many never removed them. IDs
/// repeat, so a later run inherited an earlier run's files,
/// and checks that relied on a file being absent passed the
/// wrong way.
pub(crate) struct ScratchDir {
    path: PathBuf,
}

/// A fresh [`ScratchDir`] named
/// `crabcrop-<tag>-<pid>-<n>-<nanos>`. The counter keeps
/// this process's directories apart, the nanosecond stamp
/// keeps them apart from another process that has the same
/// ID (one that ran earlier, or one in another PID
/// namespace sharing the temp dir).
pub(crate) fn scratch_dir(tag: &str) -> ScratchDir {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let name = format!("crabcrop-{tag}-{}-{n}-{nanos:08x}", std::process::id());
    ScratchDir::fresh(std::env::temp_dir().join(name))
}

impl ScratchDir {
    /// Create `path` as an empty directory. Whatever still
    /// sits under the name, left by a run that was killed
    /// before its drops ran, is removed first: the test
    /// sees only what it writes itself.
    pub(super) fn fresh(path: PathBuf) -> Self {
        if let Err(e) = std::fs::remove_dir_all(&path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            panic!("clearing the stale {}: {e}", path.display());
        }
        if let Err(e) = std::fs::create_dir(&path) {
            panic!("creating {}: {e}", path.display());
        }
        ScratchDir { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// A scratch directory reads as its path: `dir.join(..)`,
/// and `&dir` where a `&Path` is expected.
impl std::ops::Deref for ScratchDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
