//! Build and sign crabcrop URLs, per `spec/url-signing.md` in
//! the crabcrop repository. This is the same code the server
//! verifies with: one implementation, reused by every client.
//!
//! ```
//! let url = crabcrop_url::build_url(
//!     "https://images.example.com",
//!     Some(b"your-secret"),
//!     "300x200",
//!     "https://example.com/photo.jpg",
//!     None,
//! )
//! .unwrap();
//! assert_eq!(
//!     url,
//!     "https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg"
//! );
//! ```

#![forbid(unsafe_code)]

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Compute the signature segment for a remainder: the
/// lowercase hex HMAC-SHA256 of `remainder` under `secret`.
/// `remainder` is everything after the signature segment in
/// the URL, exactly as it will appear.
pub fn sign(secret: &[u8], remainder: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(remainder.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Verify a presented signature segment against a remainder,
/// in constant time. A segment that is not a 64-character hex
/// SHA-256 tag fails without panicking.
pub fn verify(secret: &[u8], signature_hex: &str, remainder: &str) -> bool {
    let Ok(expected) = hex::decode(signature_hex) else {
        return false;
    };
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(remainder.as_bytes());
    mac.verify_slice(&expected).is_ok()
}

/// Assemble a signed (or `unsafe`) crabcrop URL.
///
/// `operations` is the operation string verbatim (it may hold
/// several `/`-separated segments, `preset:` references
/// included). `expires` is a unix-seconds expiry, emitted as a
/// leading `exp:` segment. A `None` secret produces the
/// literal `unsafe` signature segment, for servers running
/// with unsafe or presets-only configurations.
/// Why a builder input can never produce a verifiable URL:
/// the remainder travels as a raw URL path, so `?` and `#`
/// would truncate it at the HTTP layer, and whitespace,
/// control, and non-ASCII characters cannot be sent as-is
/// (audit P-4). Builders fail loudly at build time instead of
/// minting a URL that 403s at fetch time.
pub fn validate_url_text(text: &str, what: &str) -> Result<(), String> {
    if let Some(c) = text
        .chars()
        .find(|c| matches!(c, '?' | '#') || !c.is_ascii() || c.is_ascii_control() || *c == ' ')
    {
        return Err(format!(
            "{what} contains a character that cannot appear in a crabcrop URL: {c:?}"
        ));
    }
    Ok(())
}

pub fn build_url(
    base: &str,
    secret: Option<&[u8]>,
    operations: &str,
    source: &str,
    expires: Option<u64>,
) -> Result<String, String> {
    validate_url_text(operations, "operations")?;
    validate_url_text(source, "source")?;
    let base = base.trim_end_matches('/');
    let mut remainder = String::new();
    if let Some(exp) = expires {
        remainder.push_str(&format!("exp:{exp}/"));
    }
    if !operations.is_empty() {
        remainder.push_str(operations);
        remainder.push('/');
    }
    remainder.push_str(source);
    let signature = match secret {
        Some(secret) => sign(secret, &remainder),
        None => "unsafe".to_string(),
    };
    Ok(format!("{base}/{signature}/{remainder}"))
}

/// Build an `srcset` attribute value: one `{width}x0` URL per
/// width, entries formatted `{url} {width}w` and joined by
/// `, `. An expiry, when given, applies to every entry.
pub fn build_srcset(
    base: &str,
    secret: Option<&[u8]>,
    source: &str,
    widths: &[u32],
    expires: Option<u64>,
) -> Result<String, String> {
    let entries = widths
        .iter()
        .map(|w| {
            let url = build_url(base, secret, &format!("{w}x0"), source, expires)?;
            Ok(format!("{url} {w}w"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(entries.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical cross-language vectors from
    /// `spec/test-vectors.json`, verified here against this
    /// crate (and, through the server crate's delegation,
    /// against the server's verification path).
    #[test]
    fn canonical_test_vectors() {
        let text = include_str!("../../../spec/test-vectors.json");
        let vectors: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert!(vectors.len() >= 5);
        for v in &vectors {
            let secret = v["secret"].as_str().unwrap();
            let remainder = v["remainder"].as_str().unwrap();
            let signature = v["signature"].as_str().unwrap();
            assert_eq!(sign(secret.as_bytes(), remainder), signature, "{remainder}");
            assert!(verify(secret.as_bytes(), signature, remainder));
            assert!(!verify(secret.as_bytes(), signature, "tampered"));
        }
    }

    #[test]
    fn builds_urls_like_the_spec() {
        // The README worked example, end to end.
        assert_eq!(
            build_url(
                "https://images.example.com/",
                Some(b"your-secret"),
                "300x200",
                "https://example.com/photo.jpg",
                None,
            )
            .unwrap(),
            "https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg"
        );
        // Expiry leads the remainder; a null secret is unsafe.
        let url = build_url(
            "https://img.example.com",
            None,
            "preset:thumb",
            "s3://media/a.jpg",
            Some(1_893_456_000),
        )
        .unwrap();
        assert_eq!(
            url,
            "https://img.example.com/unsafe/exp:1893456000/preset:thumb/s3://media/a.jpg"
        );
        // No operations: the remainder is the source alone.
        let url = build_url(
            "https://img.example.com",
            Some(b"k"),
            "",
            "https://example.com/a.jpg",
            None,
        )
        .unwrap();
        assert!(url.ends_with("/https://example.com/a.jpg"));
    }

    #[test]
    fn builders_reject_unbuildable_urls() {
        // Audit P-4: these can never verify, so building fails.
        for source in [
            "https://example.com/a.jpg?w=1",
            "https://example.com/a.jpg#frag",
            "https://example.com/my photo.jpg",
            "https://example.com/caf\u{e9}.jpg",
        ] {
            assert!(
                build_url(
                    "https://img.example.com",
                    Some(b"k"),
                    "300x200",
                    source,
                    None
                )
                .is_err(),
                "{source}"
            );
        }
        assert!(
            build_url(
                "https://img.example.com",
                Some(b"k"),
                "300x200?x",
                "https://example.com/a.jpg",
                None
            )
            .is_err()
        );
    }

    #[test]
    fn builds_srcsets() {
        let srcset = build_srcset(
            "https://img.example.com",
            None,
            "https://example.com/a.jpg",
            &[320, 640],
            None,
        )
        .unwrap();
        assert_eq!(
            srcset,
            "https://img.example.com/unsafe/320x0/https://example.com/a.jpg 320w, \
             https://img.example.com/unsafe/640x0/https://example.com/a.jpg 640w"
        );
    }
}
