//! The crabcrop server as a library: the URL parser, the
//! image pipeline, the HTTP layer, and everything they use.
//! The `crabcrop` binary (`src/main.rs`) reads the
//! environment, builds the server from these modules, and
//! runs it; the fuzz harness (`fuzz/`) and the integration
//! tests (`tests/`) import the same modules instead of
//! compiling copies of them.
//!
//! This target exists for those in-repository consumers.
//! The product contract is the URL grammar and the HTTP
//! surface in README.md, not this Rust API: module paths
//! and signatures here change whenever the server needs
//! them to, with no deprecation cycle. Clients that build
//! or sign URLs use the `crabcrop-url` crate.

#![forbid(unsafe_code)]

pub mod ai;
mod avif;
pub mod cache;
pub mod cli;
pub mod color;
pub mod crop;
pub mod debugview;
pub mod engine;
pub mod explain;
pub mod filters;
pub mod focus;
pub mod frames;
pub mod heic;
pub mod loader;
pub mod observe;
pub mod pdf;
pub mod placeholder;
pub mod provenance;
pub mod quality;
pub mod s3store;
pub mod server;
pub mod sigv4;
pub mod storage;
pub mod studio;
pub mod svg;
pub mod text;
pub mod thumbhash;
pub mod tokenizer;
pub mod toolrun;
pub mod upload;
pub mod url;
pub mod video;
