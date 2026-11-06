//! Shared resource bounds for the external tool runners
//! (issue #362): the frame extractor (frames.rs), the PDF
//! tools (pdf.rs), and the video encoder (video.rs) each
//! spawn operator-configured programs. Their wall-clock
//! timeouts kill the process group, but the children also
//! run under kernel resource limits, so a tool that
//! allocates without bound or spins on CPU meets a hard
//! stop in the kernel as well: address space is capped,
//! and CPU time is capped just past the wall-clock budget
//! (the wall clock stays the precise mechanism; the CPU
//! limit is the backstop).
//!
//! The limits are applied with `prlimit` right after the
//! spawn: the crate forbids unsafe code, so the
//! `pre_exec` route is closed. A child therefore starts
//! un-limited for the microseconds its loader needs
//! before the parent sets the limits; it cannot do
//! meaningful work of its own in that window, and the
//! wall-clock group kill covers the whole lifetime
//! regardless.

use std::time::Duration;

/// The address-space ceiling for a tool child: generous
/// headroom over the largest legitimate workload (the
/// tools render or encode at most the shared pixel and
/// decode budgets, a few hundred megabytes of buffers)
/// while still turning a runaway allocation into a tool
/// error instead of a host-level failure.
pub(crate) const TOOL_MAX_ADDRESS_SPACE: u64 = 8 * 1024 * 1024 * 1024;

/// How far past a runner's wall-clock timeout its CPU
/// limit sits. The wall clock fires first in every normal
/// shape; the CPU limit only catches a child the wall
/// clock somehow leaves running.
pub(crate) const TOOL_CPU_GRACE: Duration = Duration::from_secs(5);

/// Apply the resource limits to a spawned tool child.
/// Best effort: a limit the platform refuses is skipped
/// (the wall-clock timeout and the process-group kill
/// remain the primary bounds). Unix only; elsewhere this
/// is a no-op, as before.
pub(crate) fn limit_child(pid: u32, cpu_limit: Duration, max_address_space: u64) {
    #[cfg(unix)]
    {
        let Ok(raw) = i32::try_from(pid) else {
            return;
        };
        let Some(pid) = rustix::process::Pid::from_raw(raw) else {
            return;
        };
        let cpu_secs = cpu_limit.as_secs().max(1);
        let _ = rustix::process::prlimit(
            Some(pid),
            rustix::process::Resource::Cpu,
            rustix::process::Rlimit {
                current: Some(cpu_secs),
                maximum: Some(cpu_secs),
            },
        );
        let _ = rustix::process::prlimit(
            Some(pid),
            rustix::process::Resource::As,
            rustix::process::Rlimit {
                current: Some(max_address_space),
                maximum: Some(max_address_space),
            },
        );
    }
    #[cfg(not(unix))]
    {
        let _ = (pid, cpu_limit, max_address_space);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    #[test]
    fn cpu_limit_stops_a_spinning_child() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "while :; do :; done"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        limit_child(child.id(), Duration::from_secs(1), TOOL_MAX_ADDRESS_SPACE);
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > Duration::from_secs(6) {
                let _ = child.kill();
                panic!("the CPU limit did not stop the child");
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        assert!(!status.success(), "{status}");
        assert!(started.elapsed() < Duration::from_secs(6));
    }

    #[test]
    fn address_space_limit_stops_a_greedy_child() {
        // The greedy child allocates 3 GiB after a short
        // delay, under a 1 GiB cap set the moment it
        // spawns: the allocation fails with a memory
        // error instead of growing into the cap. The
        // delay matters: the limit lands right after
        // spawn, so the child must still be starting when
        // it does (an instant allocator can finish inside
        // the spawn-to-limit window under load, which is
        // what an earlier dd-based version of this test
        // did, flakily). python3 is the suites' usual
        // helper interpreter; skip when it is absent.
        if Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let child = Command::new("python3")
            .args([
                "-c",
                "import time; time.sleep(0.5); x = bytearray(3 * 1024**3)",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        limit_child(child.id(), Duration::from_secs(30), 1 << 30);
        let out = child.wait_with_output().unwrap();
        assert!(!out.status.success(), "{}", out.status);
    }
}
