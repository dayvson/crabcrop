//! crabcrop: on-demand image optimization.
//!
//! Every image operation is encoded in the request URL:
//!
//! ```text
//! /<signature>/<operations...>/<image-url>
//! ```
//!
//! The first path segment is the request [`url::Signature`]: either an
//! HMAC-SHA256 of the rest of the path, or the literal `unsafe` when the
//! server allows unsigned URLs.
//!
//! Configuration comes from the environment:
//!
//! - `CRABCROP_SECRET` (required): the HMAC secret.
//! - `CRABCROP_ALLOW_UNSAFE`: `1` or `true` to honor `unsafe` URLs.
//! - `CRABCROP_CACHE_MAX_MB`: result cache budget in megabytes,
//!   default 256; `0` disables the cache.
//! - `CRABCROP_ALLOWED_SOURCES`: comma-separated host suffixes the
//!   loader may fetch from; unset allows every host.
//! - `CRABCROP_BLOCK_PRIVATE_SOURCES`: `1` or `true` to refuse
//!   sources resolving to loopback, private, or link-local
//!   addresses. Off by default.
//! - `PORT`: listen port, default 3000.

mod cache;
mod crop;
mod engine;
mod filters;
mod loader;
mod observe;
mod quality;
mod server;
mod url;

use std::env;
use std::sync::Arc;

use crate::cache::Cache;
use crate::engine::Engine;
use crate::loader::{Loader, SourcePolicy};
use crate::observe::Metrics;
use crate::server::{AppState, app};

#[tokio::main]
async fn main() {
    let secret = env::var("CRABCROP_SECRET")
        .expect("CRABCROP_SECRET must be set")
        .into_bytes();
    let allow_unsafe = env::var("CRABCROP_ALLOW_UNSAFE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    let cache_mb: usize = env::var("CRABCROP_CACHE_MAX_MB")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(256);
    let allowed_sources: Vec<String> = env::var("CRABCROP_ALLOWED_SOURCES")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let block_private = env::var("CRABCROP_BLOCK_PRIVATE_SOURCES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let policy = SourcePolicy::new(allowed_sources, block_private);

    let state = AppState {
        secret,
        allow_unsafe,
        loader: Loader::with_defaults().expect("loader").with_policy(policy),
        engine: Engine,
        metrics: Arc::new(Metrics::default()),
        cache: Arc::new(Cache::new(cache_mb * 1024 * 1024)),
    };
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("bind");
    println!("crabcrop listening on port {port}");
    axum::serve(listener, app(state)).await.expect("serve");
}
