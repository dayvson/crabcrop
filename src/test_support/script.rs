//! Writing the fake tools' scripts, for every test target.
//!
//! The library's unit tests reach this through
//! `crate::test_support`. The binary's unit tests, which
//! run fake PDF tools through the boot check (issue #715),
//! are another crate and include this file by path, as they
//! include `scratch.rs`. It uses only `std` for that reason.

use std::path::Path;

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
