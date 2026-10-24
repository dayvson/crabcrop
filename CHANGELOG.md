# Changelog

All notable changes to crabcrop are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and crabcrop uses [Semantic Versioning](https://semver.org/) with one version number across every artifact (server binaries, container images, the `crabcrop-url` crate, the TypeScript and Python SDKs, the MCP server, and the Helm chart). See RELEASING.md for how a release is made.

## [Unreleased]

### Added

- Caller-supplied focal rectangle for crops: a `focal:<left>x<top>:<right>x<bottom>` segment names the source rectangle the next `crop:` must keep (center and smart crops position on it; directional alignments ignore it). The Thumbor converter now maps the `focal(...)` filter to it.
- Corner crop alignments: `crop:` accepts `top-left`, `top-right`, `bottom-left`, and `bottom-right` alongside the single-axis alignments (each corner decomposes per axis: its horizontal component positions a width crop, its vertical component a height crop). The Thumbor converter maps corner alignment pairs to them exactly instead of approximating to the vertical side.

## [0.1.0] - unreleased

The first release, staged by the Phase 6 release pipeline and not yet published to any registry.

### Added

- The server: signed image URLs (HMAC-SHA256) with resize, crop (including smart and face-aware), fit-in, pad, crop-rect, and a 45+ filter catalog; AVIF, JPEG XL, and WebP output with `Accept` q-value negotiation; canonical cache keys with memory and disk tiers; presets with purge and purge-group invalidation; meta JSON responses with placeholders (dominant color, palette, ThumbHash); optional AI features (background removal, upscale, face and class redaction, CLIP tags and alt text) behind operator-supplied ONNX models.
- Client libraries implementing the signing spec in `spec/url-signing.md`: the `crabcrop-url` Rust crate, `@crabcrop/url` for TypeScript, `crabcrop_url` for Python, framework loaders for Next.js, Nuxt, and Hydrogen, an MCP server, and the operator CLI built into the binary (sign, explain, meta, purge, purge-group, convert-thumbor).
