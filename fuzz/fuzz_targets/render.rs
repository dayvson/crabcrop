#![no_main]

use libfuzzer_sys::fuzz_target;

// The target input is a source body exactly as the loader
// fetched it, rendered with a plain resize that shrinks it:
// the scaled JPEG decode and the animated WebP frame walk
// in the render path (issue #710).
fuzz_target!(|data: &[u8]| {
    let _ = crabcrop_fuzz::check_render(data);
});
