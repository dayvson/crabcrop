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

/// Whether `id` is a well-formed key ID: 1 to 32 characters
/// of `[A-Za-z0-9_-]`. Mirrors the server grammar rule.
pub fn is_valid_key_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 32
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// The signature segment for a keyed signature: `<key-id>:`
/// followed by the hex tag [`sign`] computes. Servers verify
/// a keyed segment against the named key only, which is what
/// makes secret rotation possible without invalidating every
/// outstanding URL (see the server's rotation runbook).
pub fn sign_with_key_id(
    secret: &[u8],
    key_id: &str,
    remainder: &str,
) -> Result<String, BuildError> {
    if !is_valid_key_id(key_id) {
        return Err(BuildError::InvalidKeyId(key_id.to_string()));
    }
    Ok(format!("{key_id}:{}", sign(secret, remainder)))
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

/// The largest size dimension the server's grammar accepts,
/// per `spec/url-signing.md`.
pub const MAX_DIMENSION: u32 = 16383;

/// Why a builder refused to produce a URL. Typed, with the
/// offending value attached, so callers can react to the
/// failure instead of parsing a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// The named input ("operations" or "source") contains
    /// a character that cannot travel in a URL path.
    InvalidCharacter { what: &'static str, character: char },
    /// A srcset width is not a positive dimension within
    /// the grammar's cap. Carries the width.
    InvalidWidth(u32),
    /// A key ID is not 1 to 32 characters of `[A-Za-z0-9_-]`.
    /// Carries the offending ID.
    InvalidKeyId(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::InvalidCharacter { what, character } => write!(
                f,
                "{what} contains a character that cannot appear in a crabcrop URL: {character:?}"
            ),
            BuildError::InvalidWidth(w) => write!(
                f,
                "srcset width must be a positive integer no larger than {MAX_DIMENSION}, got {w}"
            ),
            BuildError::InvalidKeyId(id) => write!(
                f,
                "invalid key ID {id:?}: 1 to 32 characters of [A-Za-z0-9_-]"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// The purge subkey: the raw bytes of the HMAC-SHA256 of
/// the domain label `crabcrop/purge-key` under the secret.
/// Purge signatures live under this derived key, so a
/// signature minted for an image URL can never be replayed
/// against a purge endpoint.
pub fn purge_key(secret: &[u8]) -> Vec<u8> {
    // `sign` returns the hex tag; the key is its raw bytes.
    hex::decode(sign(secret, "crabcrop/purge-key")).expect("sign returns lowercase hex")
}

/// The purge signature for a target path (the request
/// remainder without its signature): HMAC-SHA256 of the
/// path under the purge subkey. This is the signature
/// segment for `POST /purge/<signature>/<path>`.
pub fn purge_signature(secret: &[u8], target: &str) -> String {
    sign(&purge_key(secret), target)
}

/// The purge-group subkey: the raw bytes of the
/// HMAC-SHA256 of the domain label
/// `crabcrop/purge-group-key` under the secret.
/// Purge-group signatures live under this derived key,
/// beside the purge key, so the two endpoints' message
/// spaces cannot overlap: a per-target purge signature
/// can never be transplanted to `/purge-group`, whatever
/// target string it was minted over (audit R2-14).
pub fn purge_group_key(secret: &[u8]) -> Vec<u8> {
    // `sign` returns the hex tag; the key is its raw bytes.
    hex::decode(sign(secret, "crabcrop/purge-group-key")).expect("sign returns lowercase hex")
}

/// The purge-group signature for a cache group: HMAC-SHA256
/// of the group name under the purge-group subkey. This is
/// the signature segment for
/// `POST /purge-group/<signature>/<group>`; the helper
/// exists so nobody has to know the signed string.
pub fn purge_group_signature(secret: &[u8], group: &str) -> String {
    sign(&purge_group_key(secret), group)
}

/// The cache entry subkey: the raw bytes of the
/// HMAC-SHA256 of the domain label `crabcrop/entry-key`
/// under the secret. Persistent cache entries are MACed
/// under this derived key (issue #190), so a forged entry
/// planted in the disk cache or the shared bucket fails
/// verification and reads as a miss. The key never
/// leaves the server: no client signature is ever
/// computed with it.
pub fn entry_key(secret: &[u8]) -> Vec<u8> {
    hex::decode(sign(secret, "crabcrop/entry-key")).expect("sign returns lowercase hex")
}

/// The upload subkey: the raw bytes of the HMAC-SHA256 of
/// the domain label `crabcrop/upload-key` under the secret.
/// Upload signatures live under this derived key, in their
/// own domain like purge signatures, so a signature minted
/// for an image URL (or a purge) can never be replayed
/// against the upload endpoint (issue #168).
pub fn upload_key(secret: &[u8]) -> Vec<u8> {
    hex::decode(sign(secret, "crabcrop/upload-key")).expect("sign returns lowercase hex")
}

/// The upload signature for an object name: HMAC-SHA256 of
/// the name under the upload subkey. This is the signature
/// segment for `PUT /upload/<signature>/<name>`.
pub fn upload_signature(secret: &[u8], name: &str) -> String {
    sign(&upload_key(secret), name)
}

/// The signed string an upload signature covers: the name
/// alone, or, for the expiring form, an `exp:<unix-seconds>`
/// segment in front of it, exactly the spelling the request
/// path carries after the signature segment. Keeping the
/// assembly in one place means signers and the server can
/// never drift on it.
pub fn upload_signed_string(name: &str, expires: Option<u64>) -> String {
    match expires {
        Some(exp) => format!("exp:{exp}/{name}"),
        None => name.to_string(),
    }
}

/// The expiring upload signature for an object name:
/// HMAC-SHA256 of `exp:<expires>/<name>` under the upload
/// subkey. The request path is
/// `PUT /upload/<signature>/exp:<expires>/<name>`; the
/// expiry sits inside the signed string, so it can neither
/// be stripped nor extended without the secret, and the
/// server refuses the upload once the expiry second is
/// reached (issue #192, audit R2-06). Delegated upload
/// tokens should always use this form: a plain
/// [`upload_signature`] token replays until the secret
/// rotates.
pub fn upload_signature_with_expiry(secret: &[u8], name: &str, expires: u64) -> String {
    sign(
        &upload_key(secret),
        &upload_signed_string(name, Some(expires)),
    )
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
pub fn validate_url_text(text: &str, what: &'static str) -> Result<(), BuildError> {
    if let Some(c) = text
        .chars()
        .find(|c| matches!(c, '?' | '#') || !c.is_ascii() || c.is_ascii_control() || *c == ' ')
    {
        return Err(BuildError::InvalidCharacter { what, character: c });
    }
    Ok(())
}

pub fn build_url(
    base: &str,
    secret: Option<&[u8]>,
    operations: &str,
    source: &str,
    expires: Option<u64>,
) -> Result<String, BuildError> {
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

/// [`build_url`] with a key ID stamped on the signature
/// segment, via [`sign_with_key_id`]. With `secret` None the
/// URL would be `unsafe` and a key ID is meaningless there,
/// so it is rejected rather than silently dropped.
pub fn build_url_with_key_id(
    base: &str,
    secret: Option<(&[u8], &str)>,
    operations: &str,
    source: &str,
    expires: Option<u64>,
) -> Result<String, BuildError> {
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
    let segment = match secret {
        Some((secret, key_id)) => sign_with_key_id(secret, key_id, &remainder)?,
        None => return Err(BuildError::InvalidKeyId(String::new())),
    };
    Ok(format!("{base}/{segment}/{remainder}"))
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
) -> Result<String, BuildError> {
    let entries = widths
        .iter()
        .map(|w| {
            if *w == 0 || *w > MAX_DIMENSION {
                return Err(BuildError::InvalidWidth(*w));
            }
            let url = build_url(base, secret, &format!("{w}x0"), source, expires)?;
            Ok(format!("{url} {w}w"))
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
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
    fn purge_signatures_match_the_published_vectors() {
        // Pinned literals: the wire scheme every deployed
        // server verifies, computed from the scheme itself.
        assert_eq!(
            purge_signature(b"test-secret", "300x200/https://example.com/a.jpg"),
            "9b58a43a984e47cbde3ec08333290acad06c48a5e9b85b99124ae725218e90e8"
        );
        assert_eq!(
            purge_group_signature(b"test-secret", "thumb"),
            "256634813bd6237179d251ba4526685f05db8a42ae056e4a04566f00042f0439"
        );
        // Domain separation: neither purge signature is an
        // image signature over a related string.
        assert_ne!(
            purge_signature(b"test-secret", "300x200/https://example.com/a.jpg"),
            sign(b"test-secret", "300x200/https://example.com/a.jpg")
        );
        // Purge-group has its own subkey (audit R2-14): no
        // per-target purge signature transplants to it, not
        // even one minted over the string the endpoint's URL
        // carries, and not one over the bare group name.
        assert_ne!(
            purge_group_signature(b"test-secret", "thumb"),
            purge_signature(b"test-secret", "purge-group/thumb")
        );
        assert_ne!(
            purge_group_signature(b"test-secret", "thumb"),
            purge_signature(b"test-secret", "thumb")
        );
        assert_ne!(purge_group_key(b"test-secret"), purge_key(b"test-secret"));
    }

    #[test]
    fn derived_subkeys_are_domain_separated() {
        // The entry MAC key is a derivation beside purge,
        // purge-group, and upload: same secret, four
        // unrelated keys, none of them the secret itself.
        let secret = b"test-secret";
        let entry = entry_key(secret);
        assert_eq!(entry.len(), 32);
        assert_ne!(entry, purge_key(secret));
        assert_ne!(entry, purge_group_key(secret));
        assert_ne!(entry, upload_key(secret));
        assert_ne!(purge_group_key(secret), upload_key(secret));
        assert_ne!(entry, secret.to_vec());
        // And it is deterministic: two derivations agree,
        // so every replica holding the secret verifies
        // the entries any other wrote.
        assert_eq!(entry, entry_key(secret));
    }

    #[test]
    fn upload_signatures_match_the_published_vectors() {
        // Pinned literals, one wire scheme across every
        // implementation (the TypeScript and Python SDKs
        // assert the same values).
        assert_eq!(
            upload_signature(b"test-secret", "heroes/cat.jpg"),
            "81d87b588203fdad32f5431c0a893bf7ddad8188f5cca123a1b057a439b34b84"
        );
        // Domain separation: an upload signature is neither
        // an image signature nor a purge signature over a
        // related string.
        assert_ne!(
            upload_signature(b"test-secret", "heroes/cat.jpg"),
            sign(b"test-secret", "heroes/cat.jpg")
        );
        assert_ne!(
            upload_signature(b"test-secret", "heroes/cat.jpg"),
            purge_signature(b"test-secret", "heroes/cat.jpg")
        );
        assert_ne!(
            upload_signature(b"test-secret", "heroes/cat.jpg"),
            upload_signature(b"test-secret", "heroes/dog.jpg")
        );
        // The expiring form: the signed string carries the
        // `exp:` segment in front of the name, pinned like
        // every other vector (the SDKs assert the same
        // value).
        assert_eq!(
            upload_signed_string("heroes/cat.jpg", None),
            "heroes/cat.jpg"
        );
        assert_eq!(
            upload_signed_string("heroes/cat.jpg", Some(1893456000)),
            "exp:1893456000/heroes/cat.jpg"
        );
        assert_eq!(
            upload_signature_with_expiry(b"test-secret", "heroes/cat.jpg", 1893456000),
            "391061e9bd262811f9bac647b5ff535f987e2478394675407f895823f45e59d2"
        );
        // An expiring token is not the plain token, and is
        // bound to both its expiry and its name.
        assert_ne!(
            upload_signature_with_expiry(b"test-secret", "heroes/cat.jpg", 1893456000),
            upload_signature(b"test-secret", "heroes/cat.jpg")
        );
        assert_ne!(
            upload_signature_with_expiry(b"test-secret", "heroes/cat.jpg", 1893456000),
            upload_signature_with_expiry(b"test-secret", "heroes/cat.jpg", 1893456001)
        );
        assert_ne!(
            upload_signature_with_expiry(b"test-secret", "heroes/cat.jpg", 1893456000),
            upload_signature_with_expiry(b"test-secret", "heroes/dog.jpg", 1893456000)
        );
    }

    #[test]
    fn builders_fail_with_typed_errors() {
        let err = build_url(
            "https://img.example.com",
            Some(b"k"),
            "300x200",
            "https://example.com/a.jpg?w=1",
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            BuildError::InvalidCharacter {
                what: "source",
                character: '?'
            }
        );
        assert_eq!(
            err.to_string(),
            "source contains a character that cannot appear in a crabcrop URL: '?'"
        );
        for w in [0, 16384] {
            assert_eq!(
                build_srcset(
                    "https://img.example.com",
                    Some(b"k"),
                    "https://example.com/a.jpg",
                    &[w],
                    None,
                )
                .unwrap_err(),
                BuildError::InvalidWidth(w)
            );
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

    #[test]
    fn keyed_signing_stamps_the_id_and_validates_it() {
        let segment = sign_with_key_id(b"test-secret", "k2", "300x200/example.com/i.jpg").unwrap();
        assert_eq!(
            segment,
            format!("k2:{}", sign(b"test-secret", "300x200/example.com/i.jpg"))
        );
        for bad in ["", "has space", "has:colon", &"x".repeat(33)] {
            assert!(matches!(
                sign_with_key_id(b"test-secret", bad, "r"),
                Err(BuildError::InvalidKeyId(_))
            ));
        }
    }

    #[test]
    fn build_url_with_key_id_places_the_segment() {
        let url = build_url_with_key_id(
            "https://img.example.com/",
            Some((b"test-secret", "k2")),
            "300x200",
            "example.com/i.jpg",
            None,
        )
        .unwrap();
        let segment = sign_with_key_id(b"test-secret", "k2", "300x200/example.com/i.jpg").unwrap();
        assert_eq!(
            url,
            format!("https://img.example.com/{segment}/300x200/example.com/i.jpg")
        );
        assert!(
            build_url_with_key_id(
                "https://img.example.com",
                None,
                "300x200",
                "example.com/i.jpg",
                None,
            )
            .is_err()
        );
    }
}
