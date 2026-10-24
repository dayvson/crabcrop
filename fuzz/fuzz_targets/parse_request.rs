#![no_main]

use libfuzzer_sys::fuzz_target;

// The target input is a full request path (signature segment
// included), exactly as the server's router receives it.
fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        crabcrop_fuzz::check(text);
    }
});
