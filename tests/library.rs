//! The library target (issue #417): integration tests and
//! the fuzz harness import the server's modules instead of
//! compiling copies of them. These pin that the pipeline is
//! reachable from outside the crate, end to end: a request
//! parsed by `crabcrop::url` renders through
//! `crabcrop::engine` on fixture bytes, and the decode
//! limits the fuzz target relies on hold at that boundary.

use crabcrop::engine::{Engine, decode_image, validate_source_image};
use crabcrop::url::parse_request;

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn a_parsed_request_renders_through_the_library() {
    let source = fixture("portrait.jpg");
    let (_, _, request) =
        parse_request("/unsafe/40x30/filters:format(png)/example.com/portrait.jpg").unwrap();
    let out = Engine.process(&source, &request.operations).unwrap();
    assert_eq!((out.width, out.height), (40, 30));
    let decoded = decode_image(&out.bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (40, 30));
}

#[test]
fn the_decode_entry_points_refuse_hostile_bytes_from_outside_the_crate() {
    // A PNG that ends after a well-formed header announcing
    // 16385x1 pixels: a trivial allocation, but one pixel
    // past the server's own dimension cap (the image crate's
    // default limits have no width cap). Both entry points
    // the fuzz target calls refuse it on the size, before
    // reading pixel data, rather than on the truncation.
    let mut ihdr = b"IHDR".to_vec();
    ihdr.extend_from_slice(&16_385u32.to_be_bytes());
    ihdr.extend_from_slice(&1u32.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut crc = flate2::Crc::new();
    crc.update(&ihdr);
    let mut bomb = b"\x89PNG\r\n\x1a\n\0\0\0\x0d".to_vec();
    bomb.extend_from_slice(&ihdr);
    bomb.extend_from_slice(&crc.sum().to_be_bytes());
    let err = decode_image(&bomb).unwrap_err().to_string();
    assert!(err.contains("size exceeds limit"), "{err}");
    let err = validate_source_image(&bomb).unwrap_err().to_string();
    assert!(err.contains("size exceeds limit"), "{err}");
    assert!(decode_image(b"").is_err());
    assert!(validate_source_image(b"not an image").is_err());
    assert!(validate_source_image(&fixture("portrait.jpg")).is_ok());
}
