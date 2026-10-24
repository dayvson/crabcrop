# crabcrop server image.
#
# Multi-stage: the builder compiles the release binary; the
# runtime stage carries only the binary and CA certificates
# (source fetching is HTTPS). The AI variant layers on top of
# this image: see Dockerfile.ai.
#
# Base images are digest-pinned (audit PL-13); refresh the
# digests deliberately when moving to a newer base.
# The bases are Debian trixie, not bookworm, on purpose:
# the AVIF decode build (below) links the system libdav1d,
# dav1d-sys requires dav1d >= 1.3, and bookworm ships
# 1.2.1. Trixie ships 1.5.x (libdav1d7).
FROM rust:1-trixie@sha256:931c2c0531a8b143e044f0f3914dfe802f15bb3d1955f0b657e9359327ac2ffc AS builder
WORKDIR /build
# pkg-config + the dav1d headers for the avif-decode build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY src src
RUN cargo build --release --locked --features avif-decode \
    && cp target/release/crabcrop /usr/local/bin/crabcrop

FROM debian:trixie-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libdav1d7 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/crabcrop /usr/local/bin/crabcrop
ENV PORT=3000
EXPOSE 3000
USER nobody
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f "http://127.0.0.1:${PORT:-3000}/health" || exit 1
ENTRYPOINT ["crabcrop"]
