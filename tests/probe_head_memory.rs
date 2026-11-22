//! What a source probe holds in memory against what it
//! charges to the source buffer budget, from outside the
//! crate and under a counting allocator. The allocator keeps
//! the live bytes of every allocation of 32 KiB or more:
//! while the probes below wait on origins that have stopped
//! sending, the only such allocations are their head buffers,
//! since the HTTP client's read buffer starts at 8 KiB and
//! grows only when a read fills it. The count is global to
//! this test binary, so the binary holds one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Duration;

use crabcrop::loader::{Loader, MediaAdmission, SourceBudget};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The global allocator of this test binary: the system
/// allocator, plus the live bytes of the allocations of at
/// least [`COUNTED`] bytes.
struct Counting;

/// The smallest allocation counted.
const COUNTED: usize = 32 * 1024;

/// The live bytes of the counted allocations.
static LIVE: AtomicIsize = AtomicIsize::new(0);

fn note(size: usize, sign: isize) {
    if size >= COUNTED {
        LIVE.fetch_add(sign * size as isize, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System` unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note(layout.size(), 1);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note(layout.size(), 1);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if !moved.is_null() {
            note(layout.size(), -1);
            note(new_size, 1);
        }
        moved
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        note(layout.size(), -1);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// How many probes each phase runs at once.
const PROBES: usize = 32;

/// An origin that answers every request with a declared
/// length of 200,000 bytes, sends the first `sent` of them,
/// all zeros, in writes of `step` bytes `every` apart, then
/// holds the connection for 30 seconds.
async fn spawn_stopping_origin(sent: usize, step: usize, every: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                let head = "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 200000\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(head.as_bytes()).await;
                let mut left = sent;
                while left > 0 {
                    let n = step.min(left);
                    if socket.write_all(&vec![0u8; n]).await.is_err() {
                        return;
                    }
                    left -= n;
                    tokio::time::sleep(every).await;
                }
                tokio::time::sleep(Duration::from_secs(30)).await;
            });
        }
    });
    format!("http://{addr}")
}

/// Run [`PROBES`] probes of `origin` at once, charged against
/// `budget` when there is one, and once they have read what
/// the origin sends and their first window is over, return
/// the counted bytes they hold and the bytes the budget is
/// charged. The probes are cancelled before it returns, and
/// give their charge back.
async fn held(origin: &str, budget: Option<&SourceBudget>) -> (isize, usize) {
    let loader = Loader::new(10 << 20, Duration::from_secs(20)).unwrap();
    let loader = Arc::new(match budget {
        Some(budget) => loader.with_source_budget(budget.clone()),
        None => loader,
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let before = LIVE.load(Ordering::Relaxed);
    let probes: Vec<_> = (0..PROBES)
        .map(|i| {
            let (loader, budget) = (loader.clone(), budget.cloned());
            let url = format!("{origin}/p{i}.png");
            let never = |_: &[u8]| false;
            tokio::spawn(async move {
                let _ = match budget {
                    Some(budget) => {
                        let charge = budget.charge(Duration::from_secs(5));
                        loader
                            .load_probe_charged(&url, MediaAdmission::default(), &never, &charge)
                            .await
                    }
                    None => {
                        loader
                            .load_probe(&url, MediaAdmission::default(), &never)
                            .await
                    }
                };
            })
        })
        .collect();
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let live = LIVE.load(Ordering::Relaxed) - before;
    let charged = budget.map_or(0, SourceBudget::in_use);
    for probe in &probes {
        probe.abort();
    }
    for probe in probes {
        let _ = probe.await;
    }
    if let Some(budget) = budget {
        assert_eq!(
            budget.in_use(),
            0,
            "cancelled probes gave their charge back"
        );
    }
    (live, charged)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_probes_hold_less_than_twice_what_they_are_charged() {
    // Origins that send one byte of a 200,000-byte body and
    // stop. Each probe reserves its head once its origin
    // answers, and once its first window is over has fallen
    // behind and is charged the one byte it holds: its head
    // buffer then takes about that byte, not a whole step of
    // the buffer's growth, and so does an uncharged probe's.
    let one_byte = spawn_stopping_origin(1, 1, Duration::ZERO).await;
    let budget = SourceBudget::new(64 << 20);
    let (live, charged) = held(&one_byte, Some(&budget)).await;
    assert_eq!(charged, PROBES, "each probe is charged the byte it holds");
    assert!(
        live <= 2 * charged as isize,
        "{PROBES} probes hold {live} bytes of head buffer, charged {charged}"
    );
    let (live, _) = held(&one_byte, None).await;
    assert!(
        live <= 2 * PROBES as isize,
        "{PROBES} uncharged probes hold {live} bytes of head buffer for one byte each"
    );
    // Origins that send 40,000 bytes in 2,000-byte writes and
    // stop: head buffers large enough to be counted, each
    // under twice the bytes its probe is charged for it.
    let forty_kb = spawn_stopping_origin(40_000, 2_000, Duration::from_millis(20)).await;
    let (live, charged) = held(&forty_kb, Some(&budget)).await;
    assert_eq!(
        charged,
        PROBES * 40_000,
        "each probe is charged what it holds"
    );
    assert!(
        live > 0 && live <= 2 * charged as isize,
        "{PROBES} probes hold {live} bytes of head buffer, charged {charged}"
    );
}
