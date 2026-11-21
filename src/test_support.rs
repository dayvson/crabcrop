//! Helpers shared by the unit tests of several modules.

use std::path::Path;

mod head_archive;
pub(crate) use head_archive::head_archive;
mod scratch;
pub(crate) use scratch::{ScratchDir, scratch_dir};
mod script;
pub(crate) use script::write_script;

/// A small deterministic generator (xorshift64*) for the
/// tests that take random steps: a seed gives the same steps
/// on every run, so a failure is named by its seed.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // Spread the seed's bits; the state is never zero.
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn step(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A number below `n`.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.step() % n as u64) as usize
    }

    /// One of `of`.
    pub(crate) fn pick<'a, T>(&mut self, of: &'a [T]) -> &'a T {
        &of[self.below(of.len())]
    }

    /// `len` random bytes.
    pub(crate) fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.step() as u8).collect()
    }
}

/// The lengths a CRC-32 from crc32fast is compared with the
/// table loops it replaced at (issue #737): every length to
/// past 1 KiB, across the sizes where the crate changes method
/// below that (16 bytes, its 128-byte fold, the 256 bytes of its
/// AVX2 loop); the 1.5 KiB where its Arm path splits into three
/// streams, which moves with the data's alignment; the 2 KiB of
/// its AVX-512 loop; a byte either side of 4 KiB, 64 KiB and
/// 1 MiB, and of the limits on the chunks crabcrop injects (the
/// caps on an embedded profile and on an XMP packet, and the
/// largest `iTXt` a plain packet makes, 22 bytes of header past
/// its cap); and megabytes, the size of an IDAT.
pub(crate) fn crc_lengths() -> Vec<usize> {
    let mut lengths: Vec<usize> = (0..=1100).chain(1530..=1560).collect();
    let edges = [
        2048,
        4096,
        65_536,
        1 << 20,
        crate::color::MAX_EMBED_ICC_BYTES,
        crate::color::MAX_XMP_BYTES,
        crate::color::MAX_XMP_BYTES + b"XML:com.adobe.xmp\0".len() + 4,
    ];
    for edge in edges {
        lengths.extend([edge - 1, edge, edge + 1]);
    }
    lengths.extend([3 << 20, (4 << 20) + 13]);
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[test]
    fn scratch_dirs_are_distinct_and_removed_on_drop() {
        let first = scratch_dir("test-support-drop");
        let second = scratch_dir("test-support-drop");
        assert_ne!(first.path(), second.path());
        assert!(first.starts_with(std::env::temp_dir()));
        assert!(first.is_dir() && second.is_dir());
        std::fs::create_dir(first.join("nested")).unwrap();
        std::fs::write(first.join("nested").join("file"), b"x").unwrap();
        let path = first.to_path_buf();
        drop(first);
        assert!(!path.exists(), "{} survived its drop", path.display());
        assert!(second.is_dir());
    }

    /// Issue #526: a name that comes round again still
    /// yields an empty directory. The stale one here holds
    /// what an earlier process left under a recycled ID in
    /// the issue's failure, a `pdfinfo` next to where a test
    /// puts its renderer.
    #[test]
    fn a_scratch_dir_starts_empty_over_a_stale_one() {
        let parent = scratch_dir("test-support-stale");
        let path = parent.join("crabcrop-boot-pdf-16683");
        std::fs::create_dir_all(path.join("nested")).unwrap();
        write_script(&path.join("pdfinfo"), "#!/bin/sh\n");
        let dir = ScratchDir::fresh(path.clone());
        assert_eq!(dir.path(), path);
        assert!(!dir.join("pdfinfo").exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }

    #[test]
    fn write_script_writes_the_exact_bytes_as_an_executable() {
        let dir = scratch_dir("test-support-exact");
        let path = dir.join("tool");
        // Quotes, a printf format, backslashes, shell
        // expansions and trailing newlines all reach the
        // file as written: the body is an argument, and
        // printf's %s copies it without interpreting it.
        let body = "#!/bin/sh\nprintf '%s|' \"$@\" '100%d' 'a\\\\b' '$HOME'\n\n\n";
        write_script(&path, body);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), body);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755);
        let out = Command::new(&path).arg("x y").output().unwrap();
        assert!(out.status.success(), "{}", out.status);
        assert_eq!(out.stdout, b"x y|100%d|a\\\\b|$HOME|");
        // A second write replaces the script, as the tests
        // that reuse a directory expect.
        write_script(&path, "#!/bin/sh\nprintf second\n");
        let out = Command::new(&path).output().unwrap();
        assert_eq!(out.stdout, b"second");
    }

    /// Write and run `iterations` scripts with `write`
    /// while three threads spawn children in a tight
    /// loop, and count the runs that fail with ETXTBSY.
    fn busy_runs_under_forks(write: fn(&Path, &str), iterations: u32) -> u32 {
        let dir = scratch_dir("test-support-stress");
        let stop = Arc::new(AtomicBool::new(false));
        let forks = Arc::new(AtomicU64::new(0));
        let forkers: Vec<_> = (0..3)
            .map(|_| {
                let stop = Arc::clone(&stop);
                let forks = Arc::clone(&forks);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let _ = Command::new("true").status();
                        forks.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        let mut busy = 0;
        for i in 0..iterations {
            let path = dir.join(format!("tool-{i}"));
            write(&path, "#!/bin/sh\nprintf ok\n");
            match Command::new(&path).stdin(Stdio::null()).output() {
                Ok(out) => assert_eq!(out.stdout, b"ok"),
                Err(err) if err.kind() == std::io::ErrorKind::ExecutableFileBusy => busy += 1,
                Err(err) => panic!("run {i}: {err}"),
            }
            std::fs::remove_file(&path).unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        for forker in forkers {
            forker.join().unwrap();
        }
        assert!(forks.load(Ordering::Relaxed) > 0);
        busy
    }

    /// The stress run of issue #523, kept out of the
    /// default run because it keeps three cores busy
    /// spawning for several seconds. Run it with
    /// `cargo test --lib test_support -- --ignored`.
    #[test]
    #[ignore = "a stress run: keeps three threads spawning for several seconds"]
    fn write_script_never_races_a_concurrent_fork() {
        assert_eq!(busy_runs_under_forks(write_script, 2000), 0);
    }
}
