//! Helpers shared by the unit tests of several modules.

use std::path::Path;

mod scratch;
pub(crate) use scratch::{ScratchDir, scratch_dir};

/// Write `body` to `path` as an executable script, for the
/// tests that stand a fake tool in for FFmpeg, pdftocairo
/// or pdfinfo.
///
/// The bytes are written by a `/bin/sh` child, never
/// through a descriptor of the test process (issue #523).
/// `cargo test` runs tests on many threads, and many of
/// them spawn tools. A child forked while this process
/// holds the script open for writing keeps a copy of that
/// descriptor until its exec, and Linux refuses to execute
/// a file that any process holds open for writing: the
/// test's own spawn then fails with ETXTBSY ("Text file
/// busy"). Closing the file before the spawn does not
/// help, since the copy outlives the close. Writing under
/// a temporary name and renaming it into place does not
/// help either: the copy refers to the file, not to its
/// name, and the rename keeps the file. A writer in its
/// own process holds no descriptor that a fork of this one
/// could copy, and it has exited, its descriptors closed,
/// before this returns.
#[cfg(unix)]
pub(crate) fn write_script(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let status = std::process::Command::new("/bin/sh")
        .args(["-c", "printf '%s' \"$2\" > \"$1\"", "write_script"])
        .arg(path)
        .arg(body)
        .stdin(std::process::Stdio::null())
        .status()
        .expect("spawn /bin/sh to write a test script");
    assert!(status.success(), "writing {}: {status}", path.display());
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Elsewhere the scripts cannot run, and there is no fork
/// to race: a plain write keeps the tests compiling.
#[cfg(not(unix))]
pub(crate) fn write_script(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
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
