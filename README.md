# crabcrop

On-demand image optimization in Rust.

Every image operation is encoded in the request URL:

```text
/<signature>/300x200/smart/filters:grayscale():quality(80)/example.com/photo.jpg
```

The server loads the source image, applies the operations, and returns the
optimized result. No build step, no pre-generated thumbnails.

## Status

Day one. The URL parser is the first primitive; the loader, engine,
transformer, filter pipeline, and HTTP server follow, one small piece at a
time, each with tests.

## Roadmap

- [x] URL signature parsing (`unsafe` vs HMAC)
- [x] URL size segment parsing (`300x200`, flips, proportional, `orig`)
- [x] Full URL grammar: crop, alignment, filters, source URL
- [x] HMAC-SHA256 request signing and verification
- [x] HTTP(S) source image loader
- [x] Image engine: decode, resize, crop, encode
- [x] Crop-to-aspect transformer
- [x] Filter pipeline: grayscale, quality, format, ...
- [x] HTTP server

## Build

```bash
cargo build
cargo test
```

## License

MIT. See [LICENSE](LICENSE).
