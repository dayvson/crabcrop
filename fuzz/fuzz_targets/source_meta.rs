#![no_main]

use libfuzzer_sys::fuzz_target;

// The target input is a source body whose embedded
// metadata (ICC profile, EXIF, XMP, PNG text chunks) the
// server extracts and may re-inject into its output.
fuzz_target!(|data: &[u8]| {
    crabcrop_fuzz::check_meta(data);
});
