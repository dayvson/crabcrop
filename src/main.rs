//! crabcrop: on-demand image optimization.
//!
//! Every image operation is encoded in the request URL:
//!
//! ```text
//! /<signature>/<operations...>/<image-url>
//! ```
//!
//! The first path segment is the request [`url::Signature`]: either an
//! HMAC-SHA1 of the rest of the path, or the literal `unsafe` when the
//! server allows unsigned URLs.

mod url;

fn main() {
    println!("crabcrop: on-demand image optimization (day one)");
}
