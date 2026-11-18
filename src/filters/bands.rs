//! The threads the filters run their row bands on (issue #739).
//!
//! A filter that splits a large frame into row bands used to
//! ask the host for its core count at every call, and to start
//! a thread per band for every pass. The bands now run on one
//! pool, started once: at boot in the server (see
//! [`start_band_pool`]), or at the first banded pass in a
//! process that boots no server. The pool has a thread for
//! each core the process may use, the count the stills lane
//! takes its permits from, and a pass still splits into at
//! most [`MAX_BANDS`] bands. The render that runs a pass
//! waits while the pool runs its bands, as a render waits
//! while an AVIF encode pool runs its encode, so the band
//! work of every render at once keeps at most a thread per
//! core busy. Each render used to start up to four threads of
//! its own per pass, so a full stills lane could put up to
//! four band threads per core on the CPU.
//!
//! A band reads nothing another band of the same pass writes,
//! so the bytes do not depend on how a pass is split or on
//! which thread runs a band: `band_proofs` holds every banded
//! filter to its one-band bytes at several pool sizes.

use std::num::NonZeroUsize;
use std::sync::OnceLock;

/// The most bands one pass splits into, whatever the pool's
/// size: the cap the band count had before the pool.
const MAX_BANDS: usize = 4;

/// Threads to run row bands on, and how many bands a pass
/// splits into.
pub(crate) struct BandPool {
    /// `None` for a single thread: every band then runs on
    /// the calling thread, one after another.
    threads: Option<rayon::ThreadPool>,
    /// The most bands one pass splits into.
    bands: usize,
}

/// The pool every filter shares, once it has started.
static SHARED: OnceLock<BandPool> = OnceLock::new();

/// Where the bands run while the shared pool's threads cannot
/// start: on the calling thread, as on a one-core host.
static CALLER: BandPool = BandPool {
    threads: None,
    bands: 1,
};

impl BandPool {
    /// A pool of `threads` threads whose passes split into at
    /// most `bands` bands.
    fn new(threads: usize, bands: usize) -> Result<BandPool, rayon::ThreadPoolBuildError> {
        let threads = if threads > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|thread| format!("filter-{thread}"))
                    .build()?,
            )
        } else {
            None
        };
        Ok(BandPool {
            threads,
            bands: bands.max(1),
        })
    }

    /// The most bands one pass splits into.
    pub(crate) fn bands(&self) -> usize {
        self.bands
    }

    /// Run `work` on every band, and return once all have run.
    /// Each band is a job of its own on the pool's threads
    /// while the calling thread waits; a lone band, or every
    /// band of a single-thread pool, runs on the calling
    /// thread instead, in order. A band that panics has its
    /// panic raised here once the other bands have ended, as
    /// a scoped thread's was, and the pool's threads live on.
    pub(crate) fn run<B, F>(&self, bands: impl IntoIterator<Item = B>, work: F)
    where
        B: Send,
        F: Fn(B) + Sync,
    {
        let mut bands = bands.into_iter();
        let Some(threads) = &self.threads else {
            bands.for_each(work);
            return;
        };
        let Some(first) = bands.next() else {
            return;
        };
        let Some(second) = bands.next() else {
            work(first);
            return;
        };
        threads.in_place_scope(|scope| {
            for band in [first, second].into_iter().chain(bands) {
                #[cfg(test)]
                POOLED_BANDS.with(|count| count.set(count.get() + 1));
                let work = &work;
                scope.spawn(move |_| work(band));
            }
        });
    }
}

/// The shared pool, started at the first call: a thread for
/// each core the process may use, read here once, and passes
/// of at most [`MAX_BANDS`] bands. Threads that cannot start
/// are not kept as the pool: that pass runs its bands on the
/// calling thread, and the next one tries again.
fn shared() -> &'static BandPool {
    if let Some(pool) = SHARED.get() {
        return pool;
    }
    let threads = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    match BandPool::new(threads, threads.min(MAX_BANDS)) {
        Ok(pool) => SHARED.get_or_init(|| pool),
        Err(_) => &CALLER,
    }
}

/// The pool the filters band on: the shared one, or in a test
/// the one [`on_test_pool`] set for the calling thread.
pub(crate) fn band_pool() -> &'static BandPool {
    #[cfg(test)]
    if let Some(pool) = TEST_POOL.with(std::cell::Cell::get) {
        return pool;
    }
    shared()
}

/// Start the pool the filters run their row bands on, unless
/// it has started (issue #739). The server calls this at
/// boot, so the host's parallelism is read and the threads
/// are up before the first render; a process that does not
/// starts the pool at its first banded pass.
pub fn start_band_pool() {
    shared();
}

#[cfg(test)]
thread_local! {
    /// The pool [`band_pool`] answers on this thread, set by
    /// [`on_test_pool`].
    static TEST_POOL: std::cell::Cell<Option<&'static BandPool>> =
        const { std::cell::Cell::new(None) };
    /// How many bands this thread has handed to a pool's
    /// threads.
    static POOLED_BANDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Run `f` with the filters on this thread banding on a pool
/// of `threads` threads that splits a pass into as many bands,
/// and return what it returns with how many bands it handed
/// to the pool's threads. A pool of one thread hands none.
/// Each size's pool is built once and kept for the test run.
#[cfg(test)]
pub(crate) fn on_test_pool<R>(threads: usize, f: impl FnOnce() -> R) -> (R, usize) {
    static POOLS: std::sync::Mutex<Vec<&'static BandPool>> = std::sync::Mutex::new(Vec::new());
    let pool = {
        let mut pools = POOLS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match pools.iter().find(|pool| pool.bands == threads) {
            Some(pool) => *pool,
            None => {
                let pool = BandPool::new(threads, threads).expect("the test pool's threads start");
                let pool: &'static BandPool = Box::leak(Box::new(pool));
                pools.push(pool);
                pool
            }
        }
    };
    struct Restore(Option<&'static BandPool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_POOL.with(|set| set.set(self.0));
        }
    }
    let _restore = Restore(TEST_POOL.with(|set| set.replace(Some(pool))));
    let before = POOLED_BANDS.with(std::cell::Cell::get);
    let out = f();
    (out, POOLED_BANDS.with(std::cell::Cell::get) - before)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::{self, ThreadId};

    /// Whether every band ran on one of a pool's threads.
    fn on_pool_threads(ran: &[(ThreadId, Option<String>)]) -> bool {
        ran.iter().all(|(_, name)| {
            name.as_deref()
                .is_some_and(|name| name.starts_with("filter-"))
        })
    }

    /// The thread each band of one pass of `bands` bands ran
    /// on, with that thread's name.
    fn threads_of_a_pass(pool: &BandPool, bands: usize) -> Vec<(ThreadId, Option<String>)> {
        let ran = Mutex::new(Vec::new());
        pool.run(0..bands, |_| {
            let me = thread::current();
            ran.lock()
                .unwrap()
                .push((me.id(), me.name().map(str::to_owned)));
        });
        ran.into_inner().unwrap()
    }

    #[test]
    fn every_pass_runs_its_bands_on_the_same_pool_threads() {
        // Issue #739: every pass started a scoped thread per
        // band, so two passes of six bands ran on twelve new
        // threads. They run on the pool's three, named for it,
        // while the calling thread waits.
        let pool = BandPool::new(3, 3).unwrap();
        let first = threads_of_a_pass(&pool, 6);
        let second = threads_of_a_pass(&pool, 6);
        assert_eq!((first.len(), second.len()), (6, 6));
        let ran: HashSet<ThreadId> = first.iter().chain(&second).map(|(id, _)| *id).collect();
        assert!(
            ran.len() <= 3,
            "two passes ran on {} threads, past the pool's 3",
            ran.len()
        );
        assert!(!ran.contains(&thread::current().id()));
        assert!(
            on_pool_threads(&first) && on_pool_threads(&second),
            "bands ran off the pool: {first:?} {second:?}"
        );
    }

    #[test]
    fn a_band_that_panics_fails_its_pass_and_the_pool_runs_the_next() {
        // The server catches a panic in a render and answers
        // that request with an error (`run_process`), and the
        // next render bands on the same pool: the panic
        // reaches the caller once the pass's other bands have
        // ended, and the pool's threads go on to the next pass.
        let pool = BandPool::new(3, 3).unwrap();
        let finished = AtomicUsize::new(0);
        let pass = catch_unwind(AssertUnwindSafe(|| {
            pool.run(0..6, |band| {
                assert_ne!(band, 2, "band 2 fails");
                finished.fetch_add(1, Ordering::Relaxed);
            })
        }));
        assert!(pass.is_err(), "the pass did not fail");
        assert_eq!(finished.load(Ordering::Relaxed), 5);
        let next = threads_of_a_pass(&pool, 6);
        assert_eq!(next.len(), 6);
        assert!(
            on_pool_threads(&next),
            "the next pass ran off the pool: {next:?}"
        );
    }
}
