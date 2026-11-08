#![no_main]

use libfuzzer_sys::fuzz_target;

// The target input is a full request path (signature
// segment included). Its filters run on the harness's
// synthetic image, and its whole operation chain runs
// through the engine on that image as a PNG (issue #476).
fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        crabcrop_fuzz::check_chain(text);
    }
});
