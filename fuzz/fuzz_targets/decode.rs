#![no_main]

use libfuzzer_sys::fuzz_target;

// The target input is a source body exactly as the loader
// fetched it: image bytes (or an SVG document) that the
// server is about to decode.
fuzz_target!(|data: &[u8]| {
    crabcrop_fuzz::check_decode(data);
});
