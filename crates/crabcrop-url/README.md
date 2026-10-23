# crabcrop-url

Build, sign, and verify [crabcrop](https://github.com/dayvson/crabcrop) URLs from Rust. The signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md) and this crate runs the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json) as its test suite. The crabcrop server verifies signatures through this same crate, so client and server cannot drift.

## Usage

```rust
use crabcrop_url::{build_srcset, build_url, sign, verify};

// A signed URL. The operation string is passed verbatim and may
// contain several segments and preset: references.
let url = build_url(
    "https://images.example.com",
    Some(b"your-secret"),
    "crop:1200x630:smart/filters:format(webp)",
    "https://example.com/photo.jpg",
    None,
);

// An expiring URL: Some(unix seconds) emits a leading exp: segment.
let expiring = build_url(
    "https://images.example.com",
    Some(b"your-secret"),
    "300x200",
    "https://example.com/photo.jpg",
    Some(1893456000),
);

// A responsive srcset: one {width}x0 URL per width.
let srcset = build_srcset(
    "https://images.example.com",
    Some(b"your-secret"),
    "https://example.com/photo.jpg",
    &[400, 800, 1200],
    None,
);

// The primitives, if you assemble or check URLs yourself.
let signature = sign(b"your-secret", "300x200/https://example.com/photo.jpg");
assert!(verify(b"your-secret", &signature, "300x200/https://example.com/photo.jpg"));
```

A `None` secret produces the literal `unsafe` signature segment, for servers running with `CRABCROP_ALLOW_UNSAFE` or in presets-only mode. Verification compares in constant time and accepts hex tags case-insensitively; builders always emit lowercase.
