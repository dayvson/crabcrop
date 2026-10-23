# crabcrop server image.
#
# Multi-stage: the builder compiles the release binary; the
# runtime stage carries only the binary and CA certificates
# (source fetching is HTTPS). The AI variant layers on top of
# this image: see Dockerfile.ai.
FROM rust:1-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY src src
RUN cargo build --release --locked \
    && cp target/release/crabcrop /usr/local/bin/crabcrop

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/crabcrop /usr/local/bin/crabcrop
ENV PORT=3000
EXPOSE 3000
USER nobody
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://127.0.0.1:3000/health || exit 1
ENTRYPOINT ["crabcrop"]
