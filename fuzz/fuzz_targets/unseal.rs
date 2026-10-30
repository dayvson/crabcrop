#![no_main]

use libfuzzer_sys::fuzz_target;

// The target input is a sealed-token signature segment as the
// server receives it, an optional `<key-id>:` prefix and all.
fuzz_target!(|data: &[u8]| {
    crabcrop_fuzz::check_seal(data);
});
