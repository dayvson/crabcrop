//! With the `zeroize` feature, the subkeys the crate derives for
//! its own signing and sealing are overwritten with zeros before
//! their buffers are freed. The allocator here looks at every
//! buffer as it is freed, and counts those that still hold the
//! subkey under watch, while a signer or opener runs.

#![cfg(feature = "zeroize")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// The system allocator, looking at each buffer it frees.
struct Watching;

/// The subkey to look for, while `WATCHING` is set.
static SUBKEY: [AtomicU8; 32] = [const { AtomicU8::new(0) }; 32];
static WATCHING: AtomicBool = AtomicBool::new(false);
/// How many freed buffers held the subkey.
static FREED_HOLDING: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call goes to the system allocator with the
// arguments it was given; `dealloc` only reads the buffer it is
// handed, which is still allocated until `System.dealloc`.
unsafe impl GlobalAlloc for Watching {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if WATCHING.load(Ordering::SeqCst) && layout.size() >= 32 {
            let subkey: [u8; 32] = std::array::from_fn(|i| SUBKEY[i].load(Ordering::SeqCst));
            // SAFETY: `ptr` is a live allocation of `layout.size()`
            // bytes until it is freed below.
            let freed = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
            if freed.windows(32).any(|w| w == subkey) {
                FREED_HOLDING.fetch_add(1, Ordering::SeqCst);
            }
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Watching = Watching;

/// One watch at a time: the subkey is the allocator's one slot.
static ONE_WATCH: Mutex<()> = Mutex::new(());

/// A signer or opener to watch: its name, the subkey it derives,
/// and a call of it.
type Watched<'a> = (&'a str, Vec<u8>, Box<dyn Fn() + 'a>);

/// How many buffers freed while `run` ran still held `subkey`.
fn freed_holding(subkey: Vec<u8>, run: impl FnOnce()) -> usize {
    let _one = ONE_WATCH.lock().unwrap_or_else(|e| e.into_inner());
    let subkey: [u8; 32] = subkey.try_into().expect("a subkey is 32 bytes");
    for (slot, byte) in SUBKEY.iter().zip(subkey) {
        slot.store(byte, Ordering::SeqCst);
    }
    FREED_HOLDING.store(0, Ordering::SeqCst);
    WATCHING.store(true, Ordering::SeqCst);
    run();
    WATCHING.store(false, Ordering::SeqCst);
    FREED_HOLDING.load(Ordering::SeqCst)
}

#[test]
fn the_subkeys_the_crate_signs_and_seals_under_are_cleared_before_they_are_freed() {
    // Each signer and opener derives its subkey into a buffer of
    // its own and drops it once done. Those buffers were freed
    // with the subkey still in them.
    let secret = b"a secret this test alone signs with";
    let remainder = "300x200/https://example.com/photo.jpg";
    let nonce = [7u8; 12];
    let token = crabcrop_url::seal_with_nonce(secret, None, &nonce, remainder).unwrap();
    let runs: [Watched; 7] = [
        (
            "purge_signature",
            crabcrop_url::purge_key(secret),
            Box::new(|| drop(crabcrop_url::purge_signature(secret, remainder))),
        ),
        (
            "purge_group_signature",
            crabcrop_url::purge_group_key(secret),
            Box::new(|| drop(crabcrop_url::purge_group_signature(secret, "thumb"))),
        ),
        (
            "upload_signature",
            crabcrop_url::upload_key(secret),
            Box::new(|| drop(crabcrop_url::upload_signature(secret, "a/b.jpg"))),
        ),
        (
            "upload_signature_with_expiry",
            crabcrop_url::upload_key(secret),
            Box::new(|| {
                drop(crabcrop_url::upload_signature_with_expiry(
                    secret,
                    "a/b.jpg",
                    1_893_456_000,
                ))
            }),
        ),
        (
            "seal",
            crabcrop_url::seal_key(secret),
            Box::new(|| drop(crabcrop_url::seal(secret, remainder).unwrap())),
        ),
        (
            "seal_with_nonce",
            crabcrop_url::seal_key(secret),
            Box::new(|| {
                drop(crabcrop_url::seal_with_nonce(secret, Some("k1"), &nonce, remainder).unwrap())
            }),
        ),
        (
            "unseal",
            crabcrop_url::seal_key(secret),
            Box::new(|| assert_eq!(crabcrop_url::unseal(secret, &token).unwrap(), remainder)),
        ),
    ];
    let held: Vec<&str> = runs
        .into_iter()
        .filter_map(|(name, subkey, run)| (freed_holding(subkey, run) > 0).then_some(name))
        .collect();
    assert!(
        held.is_empty(),
        "freed a buffer that still held the subkey: {held:?}"
    );
    // The watch itself sees a subkey freed as it is: the one the
    // caller owns, which the crate hands over uncleared.
    let held = freed_holding(crabcrop_url::purge_key(secret), || {
        drop(crabcrop_url::purge_key(secret))
    });
    assert_eq!(held, 1);
}
