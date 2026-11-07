//! Shared resource bounds for the external tool runners
//! (issue #362): the frame extractor (frames.rs), the PDF
//! tools (pdf.rs), and the video encoder (video.rs) each
//! spawn operator-configured programs. Their wall-clock
//! timeouts kill the process group, but the children also
//! run under kernel resource limits, so a tool that
//! allocates without bound or spins on CPU meets a hard
//! stop in the kernel as well: address space is capped,
//! and CPU time is capped just past the most the
//! wall-clock budget allows on this host's cores (the wall
//! clock stays the precise mechanism; the CPU limit is the
//! backstop).
//!
//! The limits are applied with `prlimit` right after the
//! spawn: the crate forbids unsafe code, so the
//! `pre_exec` route is closed. A child therefore starts
//! un-limited for the microseconds its loader needs
//! before the parent sets the limits; it cannot do
//! meaningful work of its own in that window, and the
//! wall-clock group kill covers the whole lifetime
//! regardless. `prlimit` is a Linux syscall, so the
//! kernel limits exist only there; on other platforms
//! the wall clock and the group kill are the bounds, as
//! they were before this module.

use std::time::Duration;

/// The address-space ceiling for a tool child: generous
/// headroom over the largest legitimate workload (the
/// tools render or encode at most the shared pixel and
/// decode budgets, a few hundred megabytes of buffers)
/// while still turning a runaway allocation into a tool
/// error instead of a host-level failure.
pub(crate) const TOOL_MAX_ADDRESS_SPACE: u64 = 8 * 1024 * 1024 * 1024;

/// How far past the most CPU time a run can spend inside
/// its wall-clock budget the CPU limit sits (see
/// [`tool_cpu_limit`]).
const TOOL_CPU_GRACE: Duration = Duration::from_secs(5);

/// The CPU-time limit for a tool child whose run has
/// `timeout` of wall clock. The kernel counts CPU time
/// summed over every thread of the process, and FFmpeg
/// decodes and encodes on several, so one wall-clock
/// second can cost as many CPU seconds as the host has
/// usable cores. A limit of the timeout alone killed
/// ordinary multi-threaded extractions long before their
/// wall clock ran out (issue #406), so the limit scales
/// with the cores the child may run on. The wall clock
/// then fires first in every normal shape, and the CPU
/// limit only catches a child the wall clock somehow
/// leaves running.
pub(crate) fn tool_cpu_limit(timeout: Duration) -> Duration {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    cpu_limit_for(timeout, cores)
}

/// [`tool_cpu_limit`] for a given count of usable cores.
/// One core more than counted: `available_parallelism`
/// rounds a fractional CPU quota down (1.5 CPUs count as
/// one), and a child under that quota can still spend 1.5
/// CPU seconds per wall-clock second.
fn cpu_limit_for(timeout: Duration, cores: usize) -> Duration {
    let cores = u32::try_from(cores).unwrap_or(u32::MAX).saturating_add(1);
    timeout.saturating_mul(cores).saturating_add(TOOL_CPU_GRACE)
}

/// Apply the resource limits to a spawned tool child.
/// Best effort: a limit the platform refuses is skipped
/// (the wall-clock timeout and the process-group kill
/// remain the primary bounds). Linux only: `prlimit` is
/// the one call that sets another process's limits, and
/// it exists nowhere else. Elsewhere this is a no-op, as
/// before.
pub(crate) fn limit_child(pid: u32, cpu_limit: Duration, max_address_space: u64) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
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
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = (pid, cpu_limit, max_address_space);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    // The kernel limits exist only where prlimit does;
    // elsewhere limit_child is a no-op and this child
    // would spin until the watchdog kills it.
    #[cfg(any(target_os = "linux", target_os = "android"))]
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
    fn cpu_limit_scales_with_the_usable_cores() {
        let timeout = Duration::from_secs(30);
        assert_eq!(cpu_limit_for(timeout, 1), Duration::from_secs(65));
        assert_eq!(cpu_limit_for(timeout, 4), Duration::from_secs(155));
        assert_eq!(cpu_limit_for(timeout, 64), Duration::from_secs(1955));
        // Never below what the wall clock allows on the
        // host running the test.
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!(tool_cpu_limit(timeout) > timeout * u32::try_from(cores).unwrap());
        // Absurd inputs saturate instead of overflowing.
        assert_eq!(cpu_limit_for(Duration::MAX, 4), Duration::MAX);
        assert_eq!(
            cpu_limit_for(timeout, usize::MAX),
            timeout * u32::MAX + TOOL_CPU_GRACE
        );
    }

    /// Not a test on its own: the multi-threaded spinner
    /// that `cpu_limit_lets_a_multi_threaded_child_finish`
    /// runs as its child, by re-running this test binary.
    /// The CPU limit counts every thread of one process,
    /// and no stock tool spins several threads on demand.
    #[test]
    #[ignore = "a helper the CPU limit test runs as its child"]
    fn spin_threads_helper() {
        let Ok(spec) = std::env::var("CRABCROP_TEST_SPIN") else {
            return;
        };
        let (threads, millis) = spec.split_once(':').unwrap();
        let threads: usize = threads.parse().unwrap();
        let until = std::time::Instant::now() + Duration::from_millis(millis.parse().unwrap());
        let spinners: Vec<_> = (0..threads)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut n = 0u64;
                    while std::time::Instant::now() < until {
                        n = std::hint::black_box(n.wrapping_add(1));
                    }
                })
            })
            .collect();
        for spinner in spinners {
            spinner.join().unwrap();
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn cpu_limit_lets_a_multi_threaded_child_finish() {
        // Issue #406: a child that keeps every core busy for
        // less than its wall-clock timeout must finish. With
        // a 3 s timeout, the old limit (the timeout plus the
        // grace, 8 CPU seconds) killed a 4-thread spinner on
        // a 4-core host after about 2 s of its 2.5 s run.
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let threads = cores.clamp(2, 16);
        let timeout = Duration::from_secs(3);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "toolrun::tests::spin_threads_helper",
                "--ignored",
                "--test-threads=1",
                "--quiet",
            ])
            .env("CRABCROP_TEST_SPIN", format!("{threads}:2500"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        limit_child(child.id(), tool_cpu_limit(timeout), TOOL_MAX_ADDRESS_SPACE);
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            // Generous slack over the 2.5 s run for a loaded
            // test host; a kill shows up as a failed status
            // long before this.
            if started.elapsed() > Duration::from_secs(60) {
                let _ = child.kill();
                panic!("the spinner never finished");
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        assert!(status.success(), "the CPU limit killed the child: {status}");
        // The helper really ran its spin (a renamed helper
        // would otherwise pass here without spinning).
        assert!(
            started.elapsed() >= Duration::from_secs(2),
            "the spinner ended after {:?}",
            started.elapsed()
        );
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
