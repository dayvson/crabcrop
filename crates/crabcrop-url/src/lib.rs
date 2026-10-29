//! Build, sign, seal, and verify crabcrop URLs, per
//! `spec/url-signing.md` in the crabcrop repository. This is
//! the same code the server verifies with: one
//! implementation, reused by every client.
//!
//! A crabcrop URL names an image and the operations to
//! apply to it. This crate mints and checks the four kinds
//! of authority segment such a URL can carry:
//!
//! - **Signed URLs**: an HMAC-SHA256 tag over the
//!   remainder (operations and source), optionally with an
//!   `exp:` expiry segment inside the signed text and a
//!   key ID naming the signing key, which is what makes
//!   secret rotation an overlap instead of a flag day.
//! - **Sealed URLs**: the whole remainder encrypted into
//!   an opaque `s1.` token (AES-256-GCM under a derived
//!   subkey), so the URL reveals neither the recipe nor
//!   the source. Key IDs work here exactly as for signed
//!   URLs.
//! - **Purge and upload signatures**: the same HMAC under
//!   domain-separated derived subkeys, so a signature
//!   minted for one endpoint can never be replayed
//!   against another.
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
//!
//! ```
//! // A keyed signature segment: the server verifies it
//! // against the named key only.
//! let segment = crabcrop_url::sign_with_key_id(
//!     b"your-secret",
//!     "k2026",
//!     "300x200/https://example.com/photo.jpg",
//! )
//! .unwrap();
//! assert!(segment.starts_with("k2026:"));
//! ```
//!
//! ```
//! // A sealed token round trip: opaque in the URL,
//! // openable only under the secret.
//! let token = crabcrop_url::seal(
//!     b"your-secret",
//!     "300x200/https://example.com/photo.jpg",
//! )
//! .unwrap();
//! assert!(token.starts_with("s1."));
//! assert_eq!(
//!     crabcrop_url::unseal(b"your-secret", &token).unwrap(),
//!     "300x200/https://example.com/photo.jpg"
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

/// The seal subkey: the raw bytes of the HMAC-SHA256 of
/// the domain label `crabcrop/seal-key` under the secret
/// (issue #255). Sealed tokens are encrypted under this
/// derived key, in their own domain beside purge,
/// purge-group, and upload, so no artifact crosses between
/// domains and there is no second secret to configure or
/// rotate: each keyring entry yields its own seal subkey by
/// this same derivation, and the rotation window for
/// signatures is the rotation window for tokens.
pub fn seal_key(secret: &[u8]) -> Vec<u8> {
    hex::decode(sign(secret, "crabcrop/seal-key")).expect("sign returns lowercase hex")
}

/// The version marker that opens a sealed token segment:
/// `s1.` followed by base64url (no padding) of
/// `nonce (12) || ciphertext || tag (16)`, AES-256-GCM
/// under [`seal_key`] with the fixed associated data
/// `crabcrop/sealed/1`, which binds a token to this scheme
/// and this version. The marker cannot collide with a plain
/// signature segment: a signature is 64 lowercase hex
/// characters or the literal `unsafe`, and a key ID is
/// `[A-Za-z0-9_-]`; none of them contains a dot.
pub const SEALED_PREFIX: &str = "s1.";

/// The AAD every sealed token is bound to.
const SEAL_AAD: &[u8] = b"crabcrop/sealed/1";

/// The largest remainder [`seal`] will encrypt, in bytes.
/// A recipe longer than this is a preset asking to happen;
/// the cap also bounds the token a server must decode.
pub const MAX_SEALED_REMAINDER: usize = 2048;

/// The largest token segment a server will accept, in
/// characters, checked before any decoding. Minted tokens
/// stay well under it (see [`MAX_SEALED_REMAINDER`]).
pub const MAX_SEALED_SEGMENT: usize = 4096;

/// Why sealing or opening a sealed token failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealError {
    /// The remainder is longer than
    /// [`MAX_SEALED_REMAINDER`] bytes. Carries its length.
    RemainderTooLong(usize),
    /// A key ID is not 1 to 32 characters of `[A-Za-z0-9_-]`.
    /// Carries the offending ID.
    InvalidKeyId(String),
    /// The named input ("operations" or "source") contains
    /// a character that cannot travel in a URL path, so the
    /// remainder it would seal could never be served.
    InvalidCharacter { what: &'static str, character: char },
    /// Sealing was asked for without a secret. There is no
    /// `unsafe` spelling of a token, the mirror image of how
    /// purge and upload signatures treat a null secret.
    NoSecret,
    /// The token did not open under the given secret. Every
    /// open failure (malformed body, wrong key, tampered
    /// bytes, unknown version) is this one value, on
    /// purpose: openers must never learn which stage failed,
    /// and servers answer all of them with one uniform
    /// refusal.
    OpenFailed,
}

impl std::fmt::Display for SealError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SealError::RemainderTooLong(n) => write!(
                f,
                "remainder is {n} bytes, over the {MAX_SEALED_REMAINDER}-byte sealing cap: \
                 a recipe that long belongs in a preset"
            ),
            SealError::InvalidKeyId(id) => write!(
                f,
                "invalid key ID {id:?}: 1 to 32 characters of [A-Za-z0-9_-]"
            ),
            SealError::InvalidCharacter { what, character } => write!(
                f,
                "{what} contains a character that cannot appear in a crabcrop URL: {character:?}"
            ),
            SealError::NoSecret => {
                write!(f, "sealing requires a secret: there is no unsafe token")
            }
            SealError::OpenFailed => write!(f, "the token did not open under this secret"),
        }
    }
}

impl std::error::Error for SealError {}

fn seal_cipher(secret: &[u8]) -> aes_gcm::Aes256Gcm {
    use aes_gcm::KeyInit as _;
    let key = seal_key(secret);
    let key = aes_gcm::Key::<aes_gcm::Aes256Gcm>::try_from(key.as_slice())
        .expect("an HMAC-SHA256 tag is 32 bytes");
    aes_gcm::Aes256Gcm::new(&key)
}

/// The nonce array for a 12-byte nonce, the one length
/// [`seal`] and [`open_token`] ever pass.
fn seal_nonce(
    nonce: &[u8; 12],
) -> aes_gcm::Nonce<<aes_gcm::Aes256Gcm as aes_gcm::aead::AeadCore>::NonceSize> {
    (*nonce).into()
}

fn seal_with_fixed_nonce(
    secret: &[u8],
    nonce: &[u8; 12],
    remainder: &str,
) -> Result<String, SealError> {
    use aes_gcm::aead::{Aead, Payload};
    use base64::Engine as _;
    if remainder.len() > MAX_SEALED_REMAINDER {
        return Err(SealError::RemainderTooLong(remainder.len()));
    }
    let sealed = seal_cipher(secret)
        .encrypt(
            &seal_nonce(nonce),
            Payload {
                msg: remainder.as_bytes(),
                aad: SEAL_AAD,
            },
        )
        .map_err(|_| SealError::OpenFailed)?;
    let mut body = Vec::with_capacity(12 + sealed.len());
    body.extend_from_slice(nonce);
    body.extend_from_slice(&sealed);
    Ok(format!(
        "{SEALED_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(body)
    ))
}

/// Seal a remainder into a token segment:
/// `s1.<base64url(nonce || ciphertext || tag)>`, AES-256-GCM
/// under the derived seal subkey, with a fresh random
/// 96-bit nonce from the operating system. The token is the
/// whole signature segment; stamp a key ID with
/// [`seal_with_key_id`].
pub fn seal(secret: &[u8], remainder: &str) -> Result<String, SealError> {
    let mut nonce = [0u8; 12];
    getrandom::fill(&mut nonce).map_err(|_| SealError::OpenFailed)?;
    seal_with_fixed_nonce(secret, &nonce, remainder)
}

/// [`seal`] with a caller-supplied nonce: the deterministic
/// seam the spec's test vectors use, exposed for tests and
/// for operators who need reproducible builds.
///
/// # Danger
///
/// A nonce must never repeat under one secret. Reusing a
/// nonce with AES-GCM destroys the construction's
/// confidentiality and integrity guarantees. Production
/// minting calls [`seal`], which draws a fresh random
/// nonce every time; this function exists so a test can
/// pin a nonce and compare bytes.
pub fn seal_with_nonce(
    secret: &[u8],
    key_id: Option<&str>,
    nonce: &[u8; 12],
    remainder: &str,
) -> Result<String, SealError> {
    let token = seal_with_fixed_nonce(secret, nonce, remainder)?;
    match key_id {
        Some(id) => {
            if !is_valid_key_id(id) {
                return Err(SealError::InvalidKeyId(id.to_string()));
            }
            Ok(format!("{id}:{token}"))
        }
        None => Ok(token),
    }
}

/// The token segment for a keyed seal: `<key-id>:`
/// followed by the token [`seal`] computes. Servers open a
/// keyed segment against the named key only, which is what
/// makes secret rotation work for sealed URLs exactly as
/// it does for signed ones (issue #255).
pub fn seal_with_key_id(secret: &[u8], key_id: &str, remainder: &str) -> Result<String, SealError> {
    if !is_valid_key_id(key_id) {
        return Err(SealError::InvalidKeyId(key_id.to_string()));
    }
    Ok(format!("{key_id}:{}", seal(secret, remainder)?))
}

/// Open a token body (`s1.<base64url>`, no key ID) under
/// one secret, returning the remainder it seals. Every
/// failure is [`SealError::OpenFailed`]: malformed base64,
/// a body too short to hold a nonce and a tag, a tag
/// mismatch under this secret, plaintext that is not
/// UTF-8, or a remainder over [`MAX_SEALED_REMAINDER`].
/// The cap is enforced here, not only at mint time, so
/// it holds for tokens minted by any implementation
/// (issue #264).
pub fn open_token(secret: &[u8], token: &str) -> Result<String, SealError> {
    use aes_gcm::aead::{Aead, Payload};
    use base64::Engine as _;
    let Some(body) = token.strip_prefix(SEALED_PREFIX) else {
        return Err(SealError::OpenFailed);
    };
    let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(body)
        .map_err(|_| SealError::OpenFailed)?;
    if raw.len() < 12 + 16 {
        return Err(SealError::OpenFailed);
    }
    let (nonce, sealed) = raw.split_at(12);
    let plain = seal_cipher(secret)
        .decrypt(
            &seal_nonce(nonce.try_into().expect("split_at yields the 12-byte nonce")),
            Payload {
                msg: sealed,
                aad: SEAL_AAD,
            },
        )
        .map_err(|_| SealError::OpenFailed)?;
    if plain.len() > MAX_SEALED_REMAINDER {
        return Err(SealError::OpenFailed);
    }
    String::from_utf8(plain).map_err(|_| SealError::OpenFailed)
}

/// Open a full token segment, with or without a
/// `<key-id>:` prefix, under one secret. The key ID is
/// routing metadata and is not consulted here: this
/// function's question is whether this secret opens the
/// token, the same stance [`verify`] takes toward keyed
/// signatures. Servers route by key ID through their
/// keyring instead.
pub fn unseal(secret: &[u8], segment: &str) -> Result<String, SealError> {
    let token = match segment.split_once(':') {
        Some((id, rest)) => {
            if !is_valid_key_id(id) {
                return Err(SealError::OpenFailed);
            }
            rest
        }
        None => segment,
    };
    open_token(secret, token)
}

/// Assemble a sealed crabcrop URL:
/// `{base}/[<key-id>:]s1.<token>/{friendly-name}`.
///
/// The remainder is assembled exactly as [`build_url`]
/// assembles it (expiry segment, operations, source) and
/// then sealed whole, pre-expansion, so the sealed and
/// plain forms of one render are one cache entry and one
/// purge target on the server. `name` is the cosmetic
/// friendly name: the server never parses, validates, or
/// authenticates it, and it is excluded from the token and
/// the cache key; pass `None` (or an empty name) to end
/// the URL at the token. A `None` secret is an error:
/// there is no `unsafe` spelling of a token.
#[allow(clippy::too_many_arguments)]
pub fn build_sealed_url(
    base: &str,
    secret: Option<&[u8]>,
    operations: &str,
    source: &str,
    expires: Option<u64>,
    name: Option<&str>,
    key_id: Option<&str>,
) -> Result<String, SealError> {
    validate_url_text(operations, "operations").map_err(|e| match e {
        BuildError::InvalidCharacter { what, character } => {
            SealError::InvalidCharacter { what, character }
        }
        _ => SealError::OpenFailed,
    })?;
    validate_url_text(source, "source").map_err(|e| match e {
        BuildError::InvalidCharacter { what, character } => {
            SealError::InvalidCharacter { what, character }
        }
        _ => SealError::OpenFailed,
    })?;
    let Some(secret) = secret else {
        return Err(SealError::NoSecret);
    };
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
    let segment = match key_id {
        Some(id) => seal_with_key_id(secret, id, &remainder)?,
        None => seal(secret, &remainder)?,
    };
    match name {
        Some(name) if !name.is_empty() => Ok(format!("{base}/{segment}/{name}")),
        _ => Ok(format!("{base}/{segment}")),
    }
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

/// Assemble a signed (or `unsafe`) crabcrop URL.
///
/// `operations` is the operation string verbatim (it may hold
/// several `/`-separated segments, `preset:` references
/// included). `expires` is a unix-seconds expiry, emitted as a
/// leading `exp:` segment. A `None` secret produces the
/// literal `unsafe` signature segment, for servers running
/// with unsafe or presets-only configurations.
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
        let signed: Vec<&serde_json::Value> = vectors
            .iter()
            .filter(|v| v.get("signature").is_some())
            .collect();
        assert!(signed.len() >= 5);
        for v in signed {
            let secret = v["secret"].as_str().unwrap();
            let remainder = v["remainder"].as_str().unwrap();
            let signature = v["signature"].as_str().unwrap();
            assert_eq!(sign(secret.as_bytes(), remainder), signature, "{remainder}");
            assert!(verify(secret.as_bytes(), signature, remainder));
            assert!(!verify(secret.as_bytes(), signature, "tampered"));
        }
    }

    /// The sealed vectors from `spec/test-vectors.json`
    /// (issue #255): AES-GCM is a deterministic function of
    /// (key, nonce, plaintext, AAD), so an injected nonce
    /// pins the token byte for byte, and every vector's
    /// token must open back to its remainder.
    #[test]
    fn sealed_test_vectors() {
        let text = include_str!("../../../spec/test-vectors.json");
        let vectors: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        let sealed: Vec<&serde_json::Value> = vectors
            .iter()
            .filter(|v| v["sealed"].as_bool() == Some(true))
            .collect();
        assert!(sealed.len() >= 3, "the spec carries sealed vectors");
        for v in sealed {
            let secret = v["secret"].as_str().unwrap().as_bytes();
            let key_id = v["key_id"].as_str();
            let nonce_hex = v["nonce"].as_str().unwrap();
            let nonce: [u8; 12] = hex::decode(nonce_hex).unwrap().try_into().unwrap();
            let remainder = v["remainder"].as_str().unwrap();
            let token = v["token"].as_str().unwrap();
            assert_eq!(
                seal_with_nonce(secret, key_id, &nonce, remainder).unwrap(),
                token,
                "{remainder}"
            );
            assert_eq!(unseal(secret, token).unwrap(), remainder);
            assert_eq!(unseal(b"wrong-secret", token), Err(SealError::OpenFailed));
        }
    }

    /// A tiny deterministic PRNG for the property tests:
    /// the same generator every run, so a failure
    /// reproduces from the seed printed in the assertion.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }

    /// A token minted by an implementation that ignores
    /// the cap still fails to open (issue #264): the cap
    /// is enforced on the open side, not only at mint.
    #[test]
    fn open_token_refuses_remainders_over_the_cap() {
        use aes_gcm::aead::{Aead, Payload};
        use base64::Engine as _;
        let secret = b"test-secret-for-seal-cap";
        let nonce = [7u8; 12];
        for len in [MAX_SEALED_REMAINDER, MAX_SEALED_REMAINDER + 1, 3040] {
            let remainder = "a".repeat(len);
            let sealed = seal_cipher(secret)
                .encrypt(
                    &seal_nonce(&nonce),
                    Payload {
                        msg: remainder.as_bytes(),
                        aad: SEAL_AAD,
                    },
                )
                .unwrap();
            let mut body = Vec::with_capacity(12 + sealed.len());
            body.extend_from_slice(&nonce);
            body.extend_from_slice(&sealed);
            let token = format!(
                "{SEALED_PREFIX}{}",
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(body)
            );
            if len <= MAX_SEALED_REMAINDER {
                assert_eq!(open_token(secret, &token).unwrap(), remainder);
            } else {
                assert_eq!(open_token(secret, &token), Err(SealError::OpenFailed));
            }
        }
    }

    /// Seal then open is the identity over generated
    /// remainders, including the segment kinds that ride
    /// inside a payload (`exp:`, `preset:`, `root:`) and
    /// remainders at the cap boundary (issue #255).
    #[test]
    fn seal_open_round_trip_over_generated_remainders() {
        let fragments = [
            "300x200",
            "crop:1200x630:smart",
            "filters:format(webp):quality(80)",
            "filters:text(hello,24,ffffff)",
            "exp:1893456000",
            "maxage:60",
            "preset:og-card",
            "https://example.com/photo.jpg",
            "https://cdn.example.org/a/b/c-d_e.png",
            "s3://media/photo.jpg",
            "root:photos/cat.jpg",
            "preset:thumb/filters:grayscale",
        ];
        let mut rng = XorShift(0x05ee_d255);
        for case in 0..400 {
            let parts = 1 + rng.below(4);
            let mut chosen: Vec<&str> = (0..parts)
                .map(|_| fragments[rng.below(fragments.len())])
                .collect();
            // A remainder ends with its source; put a source
            // fragment last so the shape is realistic.
            let sources = [
                "https://example.com/photo.jpg",
                "s3://media/photo.jpg",
                "root:photos/cat.jpg",
            ];
            chosen.push(sources[rng.below(sources.len())]);
            let remainder = chosen.join("/");
            let secret = format!("secret-{case}");
            let token = seal(secret.as_bytes(), &remainder).unwrap();
            assert!(
                token.starts_with(SEALED_PREFIX),
                "case {case}: {token} for {remainder}"
            );
            assert_eq!(
                unseal(secret.as_bytes(), &token).unwrap(),
                remainder,
                "case {case}"
            );
            // A keyed seal of the same remainder opens too,
            // and the key ID is outside the token body.
            let keyed = seal_with_key_id(secret.as_bytes(), "k1", &remainder).unwrap();
            assert!(keyed.starts_with("k1:s1."), "case {case}: {keyed}");
            assert_eq!(unseal(secret.as_bytes(), &keyed).unwrap(), remainder);
        }
        // At the cap, sealing works; one byte over refuses.
        let at_cap = "a".repeat(MAX_SEALED_REMAINDER);
        assert!(seal(b"s", &at_cap).is_ok());
        let over = "a".repeat(MAX_SEALED_REMAINDER + 1);
        assert_eq!(
            seal(b"s", &over),
            Err(SealError::RemainderTooLong(MAX_SEALED_REMAINDER + 1))
        );
    }

    /// Bit flips anywhere in the token body (nonce,
    /// ciphertext, tag) and any structural damage fail the
    /// open with the one opaque error (issue #255).
    #[test]
    fn open_rejects_bit_flips_and_damage() {
        let secret = b"flip-secret";
        let token = seal(secret, "300x200/https://example.com/photo.jpg").unwrap();
        let body = &token[SEALED_PREFIX.len()..];
        // Flip one character at spread positions: the first
        // body char (nonce), the middle (ciphertext), and
        // the last (tag). Base64url chars differ in bits, so
        // a swapped char is a bit flip in the decoded bytes.
        for pos in [0, body.len() / 2, body.len() - 1] {
            let mut damaged = body.to_string();
            let orig = damaged.as_bytes()[pos];
            let replacement = if orig == b'A' { 'B' } else { 'A' };
            damaged.replace_range(pos..=pos, &replacement.to_string());
            assert_eq!(
                unseal(secret, &format!("{SEALED_PREFIX}{damaged}")),
                Err(SealError::OpenFailed),
                "flip at {pos}"
            );
        }
        for damaged in [
            token[..token.len() - 1].to_string(), // truncated
            format!("{token}A"),                  // extended
            "s2.AAAA".to_string(),                // unknown version
            "s1.".to_string(),                    // empty body
            "s1.!!!".to_string(),                 // not base64url
            "s1.AAAA".to_string(),                // too short
            token.replacen("s1.", "s1", 1),       // marker damaged
            format!("bad id:{token}"),            // malformed key ID
        ] {
            assert_eq!(
                unseal(secret, &damaged),
                Err(SealError::OpenFailed),
                "{damaged}"
            );
        }
    }

    #[test]
    fn sealed_builders_and_domain_separation() {
        // The seal subkey is a fifth domain: unrelated to
        // purge, purge-group, entry, and upload subkeys, and
        // never the secret itself.
        let secret = b"test-secret";
        let key = seal_key(secret);
        assert_eq!(key.len(), 32);
        assert_ne!(key, purge_key(secret));
        assert_ne!(key, purge_group_key(secret));
        assert_ne!(key, entry_key(secret));
        assert_ne!(key, upload_key(secret));
        assert_ne!(key, secret.to_vec());
        assert_eq!(key, seal_key(secret));
        // No secret, no token.
        assert_eq!(
            build_sealed_url(
                "https://img.example.com",
                None,
                "300x200",
                "https://example.com/a.jpg",
                None,
                None,
                None,
            ),
            Err(SealError::NoSecret)
        );
        // The assembled URL: token in the signature slot,
        // the friendly name after it, nothing else.
        let url = build_sealed_url(
            "https://img.example.com/",
            Some(secret),
            "300x200",
            "https://example.com/photo.jpg",
            Some(1893456000),
            Some("photo.jpg"),
            None,
        )
        .unwrap();
        let path = url.strip_prefix("https://img.example.com/").unwrap();
        let (segment, name) = path.split_once('/').unwrap();
        assert_eq!(name, "photo.jpg");
        assert_eq!(
            unseal(secret, segment).unwrap(),
            "exp:1893456000/300x200/https://example.com/photo.jpg"
        );
        // No name: the URL ends at the token.
        let bare = build_sealed_url(
            "https://img.example.com",
            Some(secret),
            "300x200",
            "https://example.com/photo.jpg",
            None,
            None,
            Some("k2"),
        )
        .unwrap();
        let segment = bare.strip_prefix("https://img.example.com/").unwrap();
        assert!(segment.starts_with("k2:s1."), "{segment}");
        assert_eq!(
            unseal(secret, segment).unwrap(),
            "300x200/https://example.com/photo.jpg"
        );
        // Unbuildable remainders refuse at build time, like
        // the plain builders (audit P-4).
        assert!(matches!(
            build_sealed_url(
                "https://img.example.com",
                Some(secret),
                "300x200",
                "https://example.com/a.jpg?w=1",
                None,
                None,
                None,
            ),
            Err(SealError::InvalidCharacter {
                what: "source",
                character: '?'
            })
        ));
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
    fn builds_root_source_urls_signed_as_written() {
        // A short source (issue #248) is signed exactly as
        // written; the signature is the spec's canonical
        // vector for this remainder.
        assert_eq!(
            build_url(
                "https://img.example.com",
                Some(b"s3cr3t"),
                "300x200",
                "root:photos/cat.jpg",
                None,
            )
            .unwrap(),
            "https://img.example.com/47e002f609cfbfc252eb1357bbee780bbbaf4c336847610caf44617b06453824/300x200/root:photos/cat.jpg"
        );
        // The character refusals still bite a short source.
        assert!(
            build_url(
                "https://img.example.com",
                Some(b"k"),
                "300x200",
                "root:photos/cat.jpg?w=1",
                None
            )
            .is_err()
        );
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
