//! Build, sign, seal, and verify crabcrop URLs, per
//! `spec/url-signing.md` in the crabcrop repository. This is
//! the same code the server verifies with, so a Rust client
//! cannot drift from it. The TypeScript and Python SDKs are
//! separate implementations of the same spec, held to it by
//! the shared `spec/test-vectors.json`.
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
//!   minted for one endpoint verifies at no other. Each
//!   has an expiring form, with an `exp:` segment inside
//!   the signed text.
//!
//! Each subkey is the HMAC of a fixed label under the
//! secret, and every label begins with the segment
//! [`RESERVED_SEGMENT`]. No image URL may begin with that
//! segment, so every signer here refuses such a remainder
//! ([`is_reserved_remainder`]): a signature over a label
//! would be that label's subkey.
//!
//! With the `zeroize` feature, off by default, the subkeys the
//! crate derives for its own signing and sealing are overwritten
//! with zeros when it drops them, and so is the state `hmac`,
//! `sha2` and `aes-gcm` keep from a key. A subkey returned to the
//! caller ([`purge_key`] and the like) is the caller's to clear.
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

/// The first remainder segment reserved for key derivation.
/// Every derived subkey ([`purge_key`], [`purge_group_key`],
/// [`upload_key`], [`seal_key`], and the key the server
/// authenticates its stored cache entries with, which it
/// derives from `crabcrop/entry-key` itself) is the HMAC of a
/// label such as `crabcrop/purge-key` under the secret, which
/// is the image signature of that label. No image URL may
/// begin with this segment (the server refuses one with
/// `400 reserved_segment`), and every signer and builder
/// refuses a remainder that does, so no image signature is
/// ever a subkey.
pub const RESERVED_SEGMENT: &str = "crabcrop";

/// Whether `remainder` begins with [`RESERVED_SEGMENT`]: it is
/// that segment alone or starts with `crabcrop/`. The test is
/// on the bytes as written, as the HMAC is: `exp:1/crabcrop/x`
/// and `CRABCROP/x` are not reserved.
pub fn is_reserved_remainder(remainder: &str) -> bool {
    remainder.split('/').next() == Some(RESERVED_SEGMENT)
}

/// The raw HMAC-SHA256 of `message` under `key`.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

/// A subkey: the raw HMAC-SHA256 of a label under the secret.
/// Every label begins with [`RESERVED_SEGMENT`], so [`sign`]
/// refuses to compute the same value as a signature.
fn derive_subkey(secret: &[u8], label: &str) -> Vec<u8> {
    debug_assert!(is_reserved_remainder(label));
    hmac_sha256(secret, label.as_bytes()).to_vec()
}

/// A subkey this crate derives for its own signing or sealing,
/// and drops when that is done. With the `zeroize` feature, its
/// bytes are overwritten with zeros when it is dropped, before
/// its buffer is freed; without the feature the buffer is freed
/// as it is. A subkey returned to the caller ([`purge_key`] and
/// the like) is the caller's to clear.
struct Subkey(Vec<u8>);

impl Subkey {
    fn derive(secret: &[u8], label: &str) -> Self {
        Subkey(derive_subkey(secret, label))
    }
}

#[cfg(feature = "zeroize")]
impl Drop for Subkey {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.0);
    }
}

/// The label [`purge_key`] is derived from.
const PURGE_KEY_LABEL: &str = "crabcrop/purge-key";
/// The label [`purge_group_key`] is derived from.
const PURGE_GROUP_KEY_LABEL: &str = "crabcrop/purge-group-key";
/// The label [`upload_key`] is derived from.
const UPLOAD_KEY_LABEL: &str = "crabcrop/upload-key";
/// The label [`seal_key`] is derived from.
const SEAL_KEY_LABEL: &str = "crabcrop/seal-key";

/// A signature under a derived subkey: the lowercase hex
/// HMAC-SHA256 of `message`. The message space is the
/// subkey's own, so no remainder is reserved here.
fn sign_under_subkey(subkey: &[u8], message: &str) -> String {
    hex::encode(hmac_sha256(subkey, message.as_bytes()))
}

/// Compute the signature segment for a remainder: the
/// lowercase hex HMAC-SHA256 of `remainder` under `secret`.
/// `remainder` is everything after the signature segment in
/// the URL, exactly as it will appear.
///
/// A remainder that begins with [`RESERVED_SEGMENT`] is
/// refused with [`BuildError::ReservedRemainder`]: its
/// signature could be a derived subkey, and no image URL may
/// begin with that segment.
pub fn sign(secret: &[u8], remainder: &str) -> Result<String, BuildError> {
    refuse_reserved(remainder)?;
    Ok(hex::encode(hmac_sha256(secret, remainder.as_bytes())))
}

fn refuse_reserved(remainder: &str) -> Result<(), BuildError> {
    if is_reserved_remainder(remainder) {
        return Err(BuildError::ReservedRemainder(remainder.to_string()));
    }
    Ok(())
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
    Ok(format!("{key_id}:{}", sign(secret, remainder)?))
}

/// Verify a presented signature tag against a remainder, in
/// constant time. The tag is the bare hex, without a
/// `<key-id>:` prefix, in its one canonical spelling: 64
/// lowercase hex characters, as [`sign`] writes it and as the
/// server's parser takes it. Anything else, an uppercase tag
/// included, fails without panicking, so this answers what
/// the server answers. This is the raw check the server runs
/// under the secret and under each subkey, so it does not
/// refuse reserved remainders: the server refuses an image
/// remainder that begins with [`RESERVED_SEGMENT`] before it
/// verifies anything.
pub fn verify(secret: &[u8], signature_hex: &str, remainder: &str) -> bool {
    // The spelling check reads only the presented tag, never
    // the expected one, so it leaks nothing about the secret.
    if signature_hex.len() != 64
        || !signature_hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return false;
    }
    let Ok(expected) = hex::decode(signature_hex) else {
        return false;
    };
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(remainder.as_bytes());
    mac.verify_slice(&expected).is_ok()
}

/// The largest size dimension the server's grammar accepts,
/// as `docs/operations.md` in the crabcrop repository states.
/// `spec/operation-vectors.json` pins it: `16383x16383`
/// parses and `16384x10` is an `invalid_size`.
pub const MAX_DIMENSION: u32 = 16383;

/// Why a builder refused to produce a URL. Typed, with the
/// offending value attached, so callers can react to the
/// failure instead of parsing a message. A later release may
/// add a variant, so a `match` needs a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildError {
    /// The named input ("operations" or "source") contains
    /// a character that cannot travel unchanged in a URL
    /// path (the set is at [`validate_url_text`]).
    InvalidCharacter { what: &'static str, character: char },
    /// A srcset width is not a positive dimension within
    /// the grammar's cap. Carries the width.
    InvalidWidth(u32),
    /// A key ID is not 1 to 32 characters of `[A-Za-z0-9_-]`.
    /// Carries the offending ID.
    InvalidKeyId(String),
    /// The remainder begins with [`RESERVED_SEGMENT`], which
    /// no image URL may carry: its signature could be a
    /// derived subkey. Carries the remainder.
    ReservedRemainder(String),
    /// With an expiry, the named input ("operations" or
    /// "source") carries a segment that would decide the URL in
    /// the expiry's place (issue #782): an `exp:` or `maxage:`
    /// segment among the operations, or a first source segment
    /// the server reads ahead of the source, as an operation,
    /// `exp:`, `maxage:` or `preset:`. The server keeps the last
    /// `exp:` of a remainder, so such text would set the URL's
    /// lifetime whatever expiry was given, or add operations to
    /// the signed URL. Carries the input and the segment.
    ExpiryOverride { what: &'static str, segment: String },
}

/// The one message for a reserved remainder, shared by
/// [`BuildError`] and [`SealError`] and spelled the same way
/// by every first-party implementation.
fn write_reserved_remainder(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
        f,
        "remainder begins with the reserved segment \"{RESERVED_SEGMENT}\": \
         it is kept for key derivation, and no image URL may begin with it"
    )
}

/// The one message for an input that would decide an expiring
/// URL in its expiry's place, shared by [`BuildError`] and
/// [`SealError`] and spelled the same way by every first-party
/// implementation.
fn write_expiry_override(
    f: &mut std::fmt::Formatter<'_>,
    what: &str,
    segment: &str,
) -> std::fmt::Result {
    match what {
        "operations" => write!(
            f,
            "operations carries the segment \"{segment}\": with an expiry, the operations \
             may carry no exp: or maxage: segment, since the server keeps the last of each"
        ),
        _ => write!(
            f,
            "{what} begins with the segment \"{segment}\", which the server reads ahead of \
             the source: with an expiry, the source may not begin with an operation, exp:, \
             maxage: or preset: segment"
        ),
    }
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::InvalidCharacter { what, character } => {
                write_invalid_character(f, what, *character)
            }
            BuildError::InvalidWidth(w) => write!(
                f,
                "srcset width must be a positive integer no larger than {MAX_DIMENSION}, got {w}"
            ),
            BuildError::InvalidKeyId(id) => write!(
                f,
                "invalid key ID {id:?}: 1 to 32 characters of [A-Za-z0-9_-]"
            ),
            BuildError::ReservedRemainder(_) => write_reserved_remainder(f),
            BuildError::ExpiryOverride { what, segment } => write_expiry_override(f, what, segment),
        }
    }
}

impl std::error::Error for BuildError {}

/// Characters a browser rewrites in a URL path (issue #472):
/// an `<img src>` or `fetch()` URL is parsed by WHATWG rules,
/// which turn `\` into `/` and percent-encode the rest
/// (Chrome encodes `^` and `|` too). The server verifies the
/// raw path, so a URL signed over one of them answers 403
/// from a page while the same bytes verify from curl.
const BROWSER_REWRITTEN: [char; 9] = ['"', '<', '>', '\\', '^', '`', '{', '|', '}'];

/// The one message for a refused character, shared by
/// [`BuildError`] and [`SealError`]. A character a browser
/// rewrites gets the fix in the message: percent-encoded,
/// it travels unchanged and signs as written.
fn write_invalid_character(
    f: &mut std::fmt::Formatter<'_>,
    what: &str,
    character: char,
) -> std::fmt::Result {
    write!(
        f,
        "{what} contains a character that cannot appear in a crabcrop URL: {character:?}"
    )?;
    if BROWSER_REWRITTEN.contains(&character) {
        write!(
            f,
            " (browsers rewrite it; percent-encode it as %{:02X})",
            u32::from(character)
        )?;
    }
    Ok(())
}

/// The purge subkey: the raw bytes of the HMAC-SHA256 of
/// the domain label `crabcrop/purge-key` under the secret.
/// Purge signatures live under this derived key, so a
/// signature minted for an image URL does not verify at a
/// purge endpoint. That holds because [`sign`] never signs
/// the label itself (see [`RESERVED_SEGMENT`]).
pub fn purge_key(secret: &[u8]) -> Vec<u8> {
    derive_subkey(secret, PURGE_KEY_LABEL)
}

/// The purge signature for a target path (the request
/// remainder without its signature): HMAC-SHA256 of the
/// path under the purge subkey. This is the signature
/// segment for `POST /purge/<signature>/<path>`. It
/// authorizes the purge until the secret rotates; see
/// [`purge_signature_with_expiry`] for one that stops
/// working at a given second.
///
/// The server reads a leading `exp:<unix-seconds>` segment
/// of a purge path as the purge's own expiry, so a target
/// that opens with an image URL's `exp:` segment signs the
/// same bytes as the expiring form of the rest, and is
/// refused once that second is reached. An image's `exp:`
/// is not part of its cache key: drop it from the target
/// and the purge evicts the same entries.
pub fn purge_signature(secret: &[u8], target: &str) -> String {
    sign_under_subkey(&Subkey::derive(secret, PURGE_KEY_LABEL).0, target)
}

/// The signed string a purge or purge-group signature
/// covers: the target path or group name alone, or, for the
/// expiring form, an `exp:<unix-seconds>` segment in front
/// of it. It is also exactly what follows the signature
/// segment in the request path, so a URL assembled from it
/// carries the bytes its signature covers.
pub fn purge_signed_string(target: &str, expires: Option<u64>) -> String {
    match expires {
        Some(exp) => format!("exp:{exp}/{target}"),
        None => target.to_string(),
    }
}

/// The expiring purge signature for a target path:
/// HMAC-SHA256 of `exp:<expires>/<target>` under the purge
/// subkey. The request path is
/// `POST /purge/<signature>/exp:<expires>/<target>`; the
/// expiry sits inside the signed string, so it can neither
/// be stripped nor extended without the secret, and the
/// server refuses the purge once the expiry second is
/// reached, the comparison image and upload URLs use.
pub fn purge_signature_with_expiry(secret: &[u8], target: &str, expires: u64) -> String {
    sign_under_subkey(
        &purge_key(secret),
        &purge_signed_string(target, Some(expires)),
    )
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
    derive_subkey(secret, PURGE_GROUP_KEY_LABEL)
}

/// The purge-group signature for a cache group: HMAC-SHA256
/// of the group name under the purge-group subkey. This is
/// the signature segment for
/// `POST /purge-group/<signature>/<group>`; the helper
/// exists so nobody has to know the signed string. It
/// authorizes the purge until the secret rotates; see
/// [`purge_group_signature_with_expiry`] for one that stops
/// working at a given second.
pub fn purge_group_signature(secret: &[u8], group: &str) -> String {
    sign_under_subkey(&Subkey::derive(secret, PURGE_GROUP_KEY_LABEL).0, group)
}

/// The expiring purge-group signature for a cache group:
/// HMAC-SHA256 of `exp:<expires>/<group>` under the
/// purge-group subkey. The request path is
/// `POST /purge-group/<signature>/exp:<expires>/<group>`,
/// and the server refuses the purge once the expiry second
/// is reached, as for [`purge_signature_with_expiry`].
pub fn purge_group_signature_with_expiry(secret: &[u8], group: &str, expires: u64) -> String {
    sign_under_subkey(
        &purge_group_key(secret),
        &purge_signed_string(group, Some(expires)),
    )
}

/// The upload subkey: the raw bytes of the HMAC-SHA256 of
/// the domain label `crabcrop/upload-key` under the secret.
/// Upload signatures live under this derived key, in their
/// own domain like purge signatures, so a signature minted
/// for an image URL (or a purge) does not verify at the
/// upload endpoint (issue #168).
pub fn upload_key(secret: &[u8]) -> Vec<u8> {
    derive_subkey(secret, UPLOAD_KEY_LABEL)
}

/// The upload signature for an object name: HMAC-SHA256 of
/// the name under the upload subkey. This is the signature
/// segment for `PUT /upload/<signature>/<name>`.
pub fn upload_signature(secret: &[u8], name: &str) -> String {
    sign_under_subkey(&Subkey::derive(secret, UPLOAD_KEY_LABEL).0, name)
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
    derive_subkey(secret, SEAL_KEY_LABEL)
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

/// Why sealing or opening a sealed token failed. A later
/// release may add a variant, so a `match` needs a wildcard
/// arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SealError {
    /// The remainder is longer than
    /// [`MAX_SEALED_REMAINDER`] bytes. Carries its length.
    RemainderTooLong(usize),
    /// A key ID is not 1 to 32 characters of `[A-Za-z0-9_-]`.
    /// Carries the offending ID.
    InvalidKeyId(String),
    /// The named input ("operations" or "source") contains
    /// a character that cannot travel unchanged in a URL
    /// path (the set is at [`validate_url_text`]), so the
    /// remainder it would seal could never be served.
    InvalidCharacter { what: &'static str, character: char },
    /// Sealing was asked for without a secret, which only the
    /// deprecated [`build_sealed_url`] can do. There is no
    /// `unsafe` spelling of a token, the mirror image of how
    /// purge and upload signatures treat a null secret.
    NoSecret,
    /// A sealed builder assembled a remainder that begins
    /// with [`RESERVED_SEGMENT`], which no image URL may
    /// carry: the server refuses a token whose remainder
    /// does. Carries the remainder.
    ReservedRemainder(String),
    /// With an expiry, the named input ("operations" or
    /// "source") carries a segment that would decide the URL in
    /// the expiry's place, as for
    /// [`BuildError::ExpiryOverride`]: the server reads a
    /// token's remainder as it reads a plain one. Carries the
    /// input and the segment.
    ExpiryOverride { what: &'static str, segment: String },
    /// The operating system's random number generator could
    /// not draw the nonce [`seal`] needs. Carries its error,
    /// as text. Nothing was sealed; a retry may succeed.
    RandomnessFailed(String),
    /// AES-256-GCM refused to encrypt the remainder. The
    /// cipher refuses only a message far past
    /// [`MAX_SEALED_REMAINDER`], so this marks a bug, not an
    /// input to correct.
    EncryptFailed,
    /// The token did not open under the given secret. Every
    /// open failure (malformed body, wrong key, tampered
    /// bytes, unknown version) is this one value, on
    /// purpose: openers must never learn which stage failed,
    /// and servers answer all of them with one uniform
    /// refusal. Sealing never fails with it.
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
            SealError::InvalidCharacter { what, character } => {
                write_invalid_character(f, what, *character)
            }
            SealError::NoSecret => {
                write!(f, "sealing requires a secret: there is no unsafe token")
            }
            SealError::ReservedRemainder(_) => write_reserved_remainder(f),
            SealError::ExpiryOverride { what, segment } => write_expiry_override(f, what, segment),
            SealError::RandomnessFailed(cause) => write!(
                f,
                "the operating system's random number generator failed, so no nonce \
                 could be drawn: {cause}"
            ),
            SealError::EncryptFailed => write!(f, "AES-256-GCM refused to encrypt the remainder"),
            SealError::OpenFailed => write!(f, "the token did not open under this secret"),
        }
    }
}

impl std::error::Error for SealError {}

/// The cipher a token seals and opens under: AES-256-GCM keyed
/// with the seal subkey, which is dropped once the cipher holds
/// it.
fn seal_cipher(secret: &[u8]) -> aes_gcm::Aes256Gcm {
    use aes_gcm::KeyInit as _;
    let key = Subkey::derive(secret, SEAL_KEY_LABEL);
    aes_gcm::Aes256Gcm::new_from_slice(&key.0).expect("an HMAC-SHA256 tag is 32 bytes")
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
        .map_err(|_| SealError::EncryptFailed)?;
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
    getrandom::fill(&mut nonce).map_err(|e| SealError::RandomnessFailed(e.to_string()))?;
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
/// routing metadata and is not consulted here, beyond being
/// well formed: this function's question is whether this
/// secret opens the token. Servers route by key ID through
/// their keyring instead. [`verify`] is not the same: it
/// takes the bare hex tag, and a keyed signature segment
/// fails it until the `<key-id>:` prefix is split off.
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

/// A sealed crabcrop URL to build:
/// `{base}/[<key-id>:]s1.<token>/{friendly-name}`.
///
/// [`SealedUrl::new`] takes what every sealed URL has (the
/// base, the operations, and the source), the setters add
/// what some have (an expiry, a friendly name, a key ID),
/// and [`SealedUrl::build`] seals under a secret, which it
/// requires: there is no `unsafe` spelling of a token.
///
/// The remainder is assembled exactly as [`build_url`]
/// assembles it (expiry segment, operations, source) and
/// then sealed whole, pre-expansion, so the sealed and
/// plain forms of one render are one cache entry and one
/// purge target on the server.
///
/// ```
/// use crabcrop_url::{SealedUrl, unseal};
///
/// let url = SealedUrl::new(
///     "https://images.example.com",
///     "crop:1200x630:smart/filters:format(webp)",
///     "https://origin.internal/photo.jpg",
/// )
/// .expires(1893456000)
/// .name("hero.jpg")
/// .key_id("k2026")
/// .build(b"your-secret")
/// .unwrap();
/// let path = url.strip_prefix("https://images.example.com/").unwrap();
/// let (segment, name) = path.split_once('/').unwrap();
/// assert!(segment.starts_with("k2026:s1."));
/// assert_eq!(name, "hero.jpg");
/// assert_eq!(
///     unseal(b"your-secret", segment).unwrap(),
///     "exp:1893456000/crop:1200x630:smart/filters:format(webp)/https://origin.internal/photo.jpg"
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedUrl<'a> {
    base: &'a str,
    operations: &'a str,
    source: &'a str,
    expires: Option<u64>,
    name: Option<&'a str>,
    key_id: Option<&'a str>,
}

impl<'a> SealedUrl<'a> {
    /// A sealed URL for `operations` (verbatim, as for
    /// [`build_url`]) on `source`, served from `base`, with no
    /// expiry, no friendly name, and no key ID.
    #[must_use]
    pub fn new(base: &'a str, operations: &'a str, source: &'a str) -> Self {
        SealedUrl {
            base,
            operations,
            source,
            expires: None,
            name: None,
            key_id: None,
        }
    }

    /// Expire the URL at this unix second: an `exp:` segment
    /// leads the sealed remainder, and [`SealedUrl::build`]
    /// refuses operations and a source that would decide the
    /// URL in its place, as [`build_url`] does.
    #[must_use = "a setter returns the changed URL and leaves this one as it was"]
    pub fn expires(self, unix_seconds: u64) -> Self {
        SealedUrl {
            expires: Some(unix_seconds),
            ..self
        }
    }

    /// The cosmetic friendly name after the token. The server
    /// never parses, validates, or authenticates it, and it is
    /// excluded from the token and the cache key. Without one,
    /// or with an empty one, the URL ends at the token.
    #[must_use = "a setter returns the changed URL and leaves this one as it was"]
    pub fn name(self, name: &'a str) -> Self {
        SealedUrl {
            name: Some(name),
            ..self
        }
    }

    /// Stamp the token segment with a key ID, as
    /// [`seal_with_key_id`] does: the server opens it under the
    /// named key only.
    #[must_use = "a setter returns the changed URL and leaves this one as it was"]
    pub fn key_id(self, key_id: &'a str) -> Self {
        SealedUrl {
            key_id: Some(key_id),
            ..self
        }
    }

    /// Seal the remainder under `secret` and assemble the URL.
    /// It fails with the input that can never be served (a
    /// refused character, a reserved remainder, a remainder over
    /// [`MAX_SEALED_REMAINDER`], an invalid key ID), with an
    /// input that would decide the URL in its expiry's place
    /// ([`SealError::ExpiryOverride`]), or when no nonce can be
    /// drawn.
    pub fn build(&self, secret: &[u8]) -> Result<String, SealError> {
        let remainder = assemble_remainder(self.operations, self.source, self.expires)?;
        let base = self.base.trim_end_matches('/');
        let segment = match self.key_id {
            Some(id) => seal_with_key_id(secret, id, &remainder)?,
            None => seal(secret, &remainder)?,
        };
        match self.name {
            Some(name) if !name.is_empty() => Ok(format!("{base}/{segment}/{name}")),
            _ => Ok(format!("{base}/{segment}")),
        }
    }
}

/// Assemble a sealed crabcrop URL from positional arguments:
/// [`SealedUrl`] with `expires`, `name`, and `key_id` set when
/// they are `Some`. A `None` secret fails with
/// [`SealError::NoSecret`], after the inputs are checked.
#[deprecated(
    since = "0.14.0",
    note = "use `SealedUrl::new(base, operations, source)`, its setters, and `build(secret)`"
)]
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
    let Some(secret) = secret else {
        // The input checks answer first, as they always did.
        assemble_remainder(operations, source, expires)?;
        return Err(SealError::NoSecret);
    };
    let mut url = SealedUrl::new(base, operations, source);
    if let Some(expires) = expires {
        url = url.expires(expires);
    }
    if let Some(name) = name {
        url = url.name(name);
    }
    if let Some(key_id) = key_id {
        url = url.key_id(key_id);
    }
    url.build(secret)
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
    sign_under_subkey(
        &Subkey::derive(secret, UPLOAD_KEY_LABEL).0,
        &upload_signed_string(name, Some(expires)),
    )
}

/// Why a builder input can never produce a verifiable URL:
/// the remainder travels as a raw URL path, so `?` and `#`
/// would truncate it at the HTTP layer, whitespace, control,
/// and non-ASCII characters cannot be sent as-is (audit
/// P-4), and a browser rewrites `"` `<` `>` `\` `^` `` ` ``
/// `{` `|` `}` before the request leaves the page (issue
/// #472). Builders fail loudly at build time instead of
/// minting a URL that 403s at fetch time.
pub fn validate_url_text(text: &str, what: &'static str) -> Result<(), BuildError> {
    match refused_character(text) {
        Some(character) => Err(BuildError::InvalidCharacter { what, character }),
        None => Ok(()),
    }
}

/// The first character of `text` that [`validate_url_text`]
/// refuses.
fn refused_character(text: &str) -> Option<char> {
    text.chars().find(|c| {
        matches!(c, '?' | '#')
            || !c.is_ascii()
            || c.is_ascii_control()
            || *c == ' '
            || BROWSER_REWRITTEN.contains(c)
    })
}

/// Why [`assemble_remainder`] refused its inputs: the
/// refusals every builder shares, plain or sealed, which each
/// builder reports in its own error type.
enum RemainderError {
    InvalidCharacter { what: &'static str, character: char },
    Reserved(String),
    ExpiryOverride { what: &'static str, segment: String },
}

impl From<RemainderError> for BuildError {
    fn from(e: RemainderError) -> Self {
        match e {
            RemainderError::InvalidCharacter { what, character } => {
                BuildError::InvalidCharacter { what, character }
            }
            RemainderError::Reserved(remainder) => BuildError::ReservedRemainder(remainder),
            RemainderError::ExpiryOverride { what, segment } => {
                BuildError::ExpiryOverride { what, segment }
            }
        }
    }
}

impl From<RemainderError> for SealError {
    fn from(e: RemainderError) -> Self {
        match e {
            RemainderError::InvalidCharacter { what, character } => {
                SealError::InvalidCharacter { what, character }
            }
            RemainderError::Reserved(remainder) => SealError::ReservedRemainder(remainder),
            RemainderError::ExpiryOverride { what, segment } => {
                SealError::ExpiryOverride { what, segment }
            }
        }
    }
}

/// The prefixes of the segments the server reads ahead of the
/// source (`parse_operations_with_expiry` and `parse_operation`
/// in the server's `src/url.rs`): the `exp:` and `maxage:`
/// metadata, a `preset:` reference, and every operation that
/// opens with a name. A size has none (see [`size_shaped`]).
const LEADING_PREFIXES: [&str; 13] = [
    "exp:",
    "maxage:",
    "preset:",
    "debug:",
    "frame:",
    "page:",
    "crop-rect:",
    "focalpoint:",
    "focal:",
    "crop:",
    "pad:",
    "fit-in:",
    "filters:",
];

/// The server's size shape: an `x` joining two parts, at least
/// one of them a dimension spelling (digits, `orig`, or
/// nothing, after an optional `-`), and no `.` or `:` in the
/// segment. Every size the server parses has it, and the
/// server reads a segment with it that does not parse as a
/// malformed size, never as source text.
fn size_shaped(segment: &str) -> bool {
    let Some((left, right)) = segment.split_once('x') else {
        return false;
    };
    if segment.contains('.') || segment.contains(':') {
        return false;
    }
    let spelling = |part: &str| {
        let digits = part.strip_prefix('-').unwrap_or(part);
        digits == "orig" || digits.bytes().all(|b| b.is_ascii_digit())
    };
    spelling(left) || spelling(right)
}

/// Whether the server reads `segment` ahead of the source:
/// as metadata, a preset reference or an operation, valid or
/// not. The first segment that is none of these starts the
/// source.
fn read_before_source(segment: &str) -> bool {
    LEADING_PREFIXES
        .iter()
        .any(|prefix| segment.starts_with(prefix))
        || size_shaped(segment)
}

/// With an expiry, the input and segment that would decide
/// the URL in the expiry's place (issue #782): an `exp:` or
/// `maxage:` segment anywhere in the operations, or a first
/// source segment the server reads ahead of the source. The
/// server keeps the last `exp:` of a remainder, so either
/// would let the text a caller passed outlive the expiry it
/// gave, and a source read as operations would sign them too.
fn expiry_override<'a>(operations: &'a str, source: &'a str) -> Option<(&'static str, &'a str)> {
    if let Some(segment) = operations
        .split('/')
        .find(|s| s.starts_with("exp:") || s.starts_with("maxage:"))
    {
        return Some(("operations", segment));
    }
    let first = source.split('/').next().unwrap_or(source);
    read_before_source(first).then_some(("source", first))
}

/// The remainder every builder signs or seals: the expiry
/// segment, the operations, and the source, joined by `/`.
/// Inputs that can never verify are refused here, plain or
/// sealed, with a secret or without: a character
/// [`validate_url_text`] refuses, and a remainder that
/// begins with [`RESERVED_SEGMENT`], which the server
/// refuses whatever the signature. With an expiry, so is
/// text that would decide the URL in the expiry's place
/// (see [`BuildError::ExpiryOverride`]).
fn assemble_remainder(
    operations: &str,
    source: &str,
    expires: Option<u64>,
) -> Result<String, RemainderError> {
    for (text, what) in [(operations, "operations"), (source, "source")] {
        if let Some(character) = refused_character(text) {
            return Err(RemainderError::InvalidCharacter { what, character });
        }
    }
    if let Some((what, segment)) = expires.and_then(|_| expiry_override(operations, source)) {
        return Err(RemainderError::ExpiryOverride {
            what,
            segment: segment.to_string(),
        });
    }
    let mut remainder = String::new();
    if let Some(exp) = expires {
        remainder.push_str(&format!("exp:{exp}/"));
    }
    if !operations.is_empty() {
        remainder.push_str(operations);
        remainder.push('/');
    }
    remainder.push_str(source);
    if is_reserved_remainder(&remainder) {
        return Err(RemainderError::Reserved(remainder));
    }
    Ok(remainder)
}

/// Assemble a signed (or `unsafe`) crabcrop URL.
///
/// `operations` is the operation string verbatim (it may hold
/// several `/`-separated segments, `preset:` references
/// included). `expires` is a unix-seconds expiry, emitted as a
/// leading `exp:` segment. With one, the operations may carry
/// no `exp:` or `maxage:` segment, and the source may not
/// begin with a segment the server reads as an operation,
/// `exp:`, `maxage:` or `preset:`: the server keeps the last
/// `exp:` of a URL, so either is refused, with
/// [`BuildError::ExpiryOverride`]. A `None` secret produces the
/// literal `unsafe` signature segment, for servers running
/// with unsafe or presets-only configurations. A remainder
/// that begins with [`RESERVED_SEGMENT`] is refused either
/// way, with [`BuildError::ReservedRemainder`].
pub fn build_url(
    base: &str,
    secret: Option<&[u8]>,
    operations: &str,
    source: &str,
    expires: Option<u64>,
) -> Result<String, BuildError> {
    let remainder = assemble_remainder(operations, source, expires)?;
    let base = base.trim_end_matches('/');
    let signature = match secret {
        Some(secret) => sign(secret, &remainder)?,
        None => "unsafe".to_string(),
    };
    Ok(format!("{base}/{signature}/{remainder}"))
}

/// [`build_url`] with a key ID stamped on the signature
/// segment, via [`sign_with_key_id`]. `secret` is the secret
/// and the ID of the key it is: a keyed URL is always signed,
/// since an `unsafe` URL has no key to name.
pub fn build_url_with_key_id(
    base: &str,
    secret: (&[u8], &str),
    operations: &str,
    source: &str,
    expires: Option<u64>,
) -> Result<String, BuildError> {
    let (secret, key_id) = secret;
    let remainder = assemble_remainder(operations, source, expires)?;
    let base = base.trim_end_matches('/');
    let segment = sign_with_key_id(secret, key_id, &remainder)?;
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

    /// The cross-language vectors, `spec/test-vectors.json` in
    /// the crabcrop repository. The package carries its own
    /// copy, so its tests build from the published crate,
    /// which ships no `spec/` directory (issue #712);
    /// [`vendored_vectors_are_the_spec_file`] holds the copy to
    /// the spec's file in the repository.
    const VECTORS: &str = include_str!("../tests/test-vectors.json");

    /// The cache entry subkey's label, which the server
    /// derives its entry key from: a signer refuses it like
    /// every other label.
    const ENTRY_KEY_LABEL: &str = "crabcrop/entry-key";

    /// In the repository, the vendored vectors are the spec's
    /// file byte for byte. A published package has no spec to
    /// compare with, and its copy is the one the release
    /// tested. Refresh the copy with
    /// `cp spec/test-vectors.json crates/crabcrop-url/tests/`.
    #[test]
    fn vendored_vectors_are_the_spec_file() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if !repo.join("spec/url-signing.md").is_file() {
            return;
        }
        let spec = std::fs::read_to_string(repo.join("spec/test-vectors.json"))
            .expect("the repository's spec/test-vectors.json");
        assert!(
            spec == VECTORS,
            "crates/crabcrop-url/tests/test-vectors.json differs from spec/test-vectors.json: \
             cp spec/test-vectors.json crates/crabcrop-url/tests/"
        );
    }

    /// The canonical cross-language vectors from
    /// `spec/test-vectors.json`, verified here against this
    /// crate (and, through the server crate's delegation,
    /// against the server's verification path).
    #[test]
    fn canonical_test_vectors() {
        let text = VECTORS;
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
            assert_eq!(
                sign(secret.as_bytes(), remainder).unwrap(),
                signature,
                "{remainder}"
            );
            assert!(verify(secret.as_bytes(), signature, remainder));
            assert!(!verify(secret.as_bytes(), signature, "tampered"));
            // One canonical spelling, as the server's parser
            // takes it (issue #712): the uppercase spelling of
            // the same tag is well-formed hex, and it fails.
            let upper = signature.to_uppercase();
            assert_ne!(upper, signature);
            assert!(!verify(secret.as_bytes(), &upper, remainder), "{remainder}");
        }
    }

    /// `verify` takes the canonical tag and nothing else: 64
    /// lowercase hex characters, the bare tag with no key ID
    /// (issue #712). Mixed case, a key-ID prefix, a short or
    /// long tag, and non-hex all fail, without panicking.
    #[test]
    fn verify_takes_only_the_canonical_tag() {
        let secret = b"verify-spelling-secret";
        let remainder = "300x200/https://example.com/a.jpg";
        let tag = sign(secret, remainder).unwrap();
        assert!(verify(secret, &tag, remainder));
        let first_letter = tag.find(|c: char| c.is_ascii_lowercase()).unwrap();
        let mut mixed = tag.clone();
        mixed.replace_range(
            first_letter..=first_letter,
            &tag[first_letter..=first_letter].to_uppercase(),
        );
        for bad in [
            mixed,
            tag.to_uppercase(),
            format!("k2:{tag}"),
            tag[..62].to_string(),
            format!("{tag}00"),
            format!("{}zz", &tag[..62]),
            String::new(),
            "é".repeat(32),
        ] {
            assert!(!verify(secret, &bad, remainder), "{bad}");
        }
    }

    /// The sealed vectors from `spec/test-vectors.json`
    /// (issue #255): AES-GCM is a deterministic function of
    /// (key, nonce, plaintext, AAD), so an injected nonce
    /// pins the token byte for byte, and every vector's
    /// token must open back to its remainder.
    #[test]
    fn sealed_test_vectors() {
        let text = VECTORS;
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

    /// The purge vectors from `spec/test-vectors.json`: each
    /// endpoint's helper, plain or expiring, reproduces the
    /// signature, `purge_signed_string` the signed string,
    /// which is also what follows the signature segment in
    /// the request path, and the expiring form is the plain
    /// signature over that string under the endpoint's own
    /// subkey.
    #[test]
    fn purge_test_vectors() {
        type Plain = fn(&[u8], &str) -> String;
        type Expiring = fn(&[u8], &str, u64) -> String;
        type Subkey = fn(&[u8]) -> Vec<u8>;
        let vectors: Vec<serde_json::Value> = serde_json::from_str(VECTORS).unwrap();
        let purges: Vec<&serde_json::Value> = vectors
            .iter()
            .filter(|v| v.get("purge_signature").is_some())
            .collect();
        assert!(purges.len() >= 6, "the spec carries purge vectors");
        let mut expiring = 0;
        for v in purges {
            let description = v["description"].as_str().unwrap();
            let secret = v["secret"].as_str().unwrap().as_bytes();
            let target = v["target"].as_str().unwrap();
            let expires = v.get("expires").map(|e| e.as_u64().unwrap());
            let signed = v["signed"].as_str().unwrap();
            let expected = v["purge_signature"].as_str().unwrap();
            let (plain, with_expiry, subkey): (Plain, Expiring, Subkey) =
                match v["endpoint"].as_str().unwrap() {
                    "purge" => (purge_signature, purge_signature_with_expiry, purge_key),
                    "purge-group" => (
                        purge_group_signature,
                        purge_group_signature_with_expiry,
                        purge_group_key,
                    ),
                    other => panic!("{description}: unknown endpoint {other}"),
                };
            assert_eq!(
                purge_signed_string(target, expires),
                signed,
                "{description}"
            );
            let minted = match expires {
                Some(exp) => {
                    expiring += 1;
                    with_expiry(secret, target, exp)
                }
                None => plain(secret, target),
            };
            assert_eq!(minted, expected, "{description}");
            assert_eq!(plain(secret, signed), expected, "{description}");
            assert!(verify(&subkey(secret), expected, signed), "{description}");
            assert!(!verify(secret, expected, signed), "{description}");
        }
        assert!(expiring >= 3, "the spec carries expiring purge vectors");
    }

    /// The refusal vectors from `spec/test-vectors.json`
    /// (issue #472): every builder refuses each input with
    /// the typed error naming the input and the character,
    /// signed or not, plain or sealed, and a character a
    /// browser rewrites carries its percent-encoding in the
    /// message.
    #[test]
    fn refusal_test_vectors() {
        let text = VECTORS;
        let vectors: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        let refusals: Vec<&serde_json::Value> = vectors
            .iter()
            .filter(|v| v.get("refuses").is_some())
            .collect();
        assert!(refusals.len() >= 17, "the spec carries refusal vectors");
        let base = "https://img.example.com";
        for v in refusals {
            let operations = v["operations"].as_str().unwrap();
            let source = v["source"].as_str().unwrap();
            let what = v["refuses"]["input"].as_str().unwrap();
            let mut chars = v["refuses"]["character"].as_str().unwrap().chars();
            let character = chars.next().unwrap();
            assert!(chars.next().is_none());
            let description = v["description"].as_str().unwrap();
            for secret in [Some(&b"s3cr3t"[..]), None] {
                let err = build_url(base, secret, operations, source, None).unwrap_err();
                assert_eq!(
                    err,
                    BuildError::InvalidCharacter {
                        what: if what == "source" {
                            "source"
                        } else {
                            "operations"
                        },
                        character
                    },
                    "{description}"
                );
                let message = err.to_string();
                assert!(
                    message.starts_with(&format!("{what} contains")),
                    "{message}"
                );
                if BROWSER_REWRITTEN.contains(&character) {
                    let hint = format!("percent-encode it as %{:02X})", u32::from(character));
                    assert!(message.ends_with(&hint), "{message}");
                } else {
                    assert!(!message.contains("percent-encode"), "{message}");
                }
            }
            assert!(matches!(
                build_url_with_key_id(base, (b"s3cr3t", "k2"), operations, source, None),
                Err(BuildError::InvalidCharacter { character: c, .. }) if c == character
            ));
            assert!(
                matches!(
                    SealedUrl::new(base, operations, source).build(b"s3cr3t"),
                    Err(SealError::InvalidCharacter { character: c, .. }) if c == character
                ),
                "{description}"
            );
            #[allow(deprecated)]
            let shim = build_sealed_url(base, None, operations, source, None, None, None);
            assert!(
                matches!(
                    shim,
                    Err(SealError::InvalidCharacter { character: c, .. }) if c == character
                ),
                "{description}"
            );
            if what == "source" {
                assert!(matches!(
                    build_srcset(base, Some(b"s3cr3t"), source, &[320], None),
                    Err(BuildError::InvalidCharacter { character: c, .. }) if c == character
                ));
            }
        }
    }

    /// The reserved vectors from `spec/test-vectors.json`:
    /// `sign` and every builder refuse a remainder that
    /// begins with the reserved segment, signed or not, keyed
    /// or not, plain or sealed, with the one typed error and
    /// the one message every implementation spells alike.
    #[test]
    fn reserved_test_vectors() {
        let text = VECTORS;
        let vectors: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        let reserved: Vec<&serde_json::Value> = vectors
            .iter()
            .filter(|v| v["reserved"].as_bool() == Some(true))
            .collect();
        assert!(reserved.len() >= 9, "the spec carries reserved vectors");
        let base = "https://img.example.com";
        let secret = &b"s3cr3t"[..];
        let nonce = [7u8; 12];
        for v in reserved {
            let operations = v["operations"].as_str().unwrap();
            let source = v["source"].as_str().unwrap();
            let remainder = v["remainder"].as_str().unwrap();
            let description = v["description"].as_str().unwrap();
            assert!(is_reserved_remainder(remainder), "{description}");
            let refused = BuildError::ReservedRemainder(remainder.to_string());
            assert_eq!(
                sign(secret, remainder),
                Err(refused.clone()),
                "{description}"
            );
            assert!(
                refused
                    .to_string()
                    .contains("reserved segment \"crabcrop\""),
                "{refused}"
            );
            assert_eq!(
                sign_with_key_id(secret, "k2", remainder),
                Err(refused.clone()),
                "{description}"
            );
            for secret in [Some(secret), None] {
                assert_eq!(
                    build_url(base, secret, operations, source, None),
                    Err(refused.clone()),
                    "{description}"
                );
            }
            assert_eq!(
                build_url_with_key_id(base, (secret, "k2"), operations, source, None),
                Err(refused.clone()),
                "{description}"
            );
            let sealed = SealError::ReservedRemainder(remainder.to_string());
            assert_eq!(sealed.to_string(), refused.to_string());
            let url = SealedUrl::new(base, operations, source);
            for url in [url, url.key_id("k2"), url.name("a.jpg")] {
                assert_eq!(url.build(secret), Err(sealed.clone()), "{description}");
            }
            for secret in [Some(secret), None] {
                #[allow(deprecated)]
                let shim = build_sealed_url(base, secret, operations, source, None, None, None);
                assert_eq!(shim, Err(sealed.clone()), "{description}");
            }
            // The sealing primitive encrypts under the seal
            // subkey and computes no HMAC under the secret, so
            // it is not where the reserved segment is policed;
            // like characters, the builders police it.
            let token = seal_with_nonce(secret, None, &nonce, remainder).unwrap();
            assert_eq!(unseal(secret, &token).unwrap(), remainder);
        }
    }

    /// The message an expiry refusal carries, as every
    /// first-party implementation spells it.
    fn expiry_override_message(what: &str, segment: &str) -> String {
        if what == "operations" {
            format!(
                "operations carries the segment \"{segment}\": with an expiry, the operations \
                 may carry no exp: or maxage: segment, since the server keeps the last of each"
            )
        } else {
            format!(
                "source begins with the segment \"{segment}\", which the server reads ahead of \
                 the source: with an expiry, the source may not begin with an operation, exp:, \
                 maxage: or preset: segment"
            )
        }
    }

    /// The expiry vectors from `spec/test-vectors.json` (issue
    /// #782). With an expiry, every builder refuses operations
    /// that carry an `exp:` or `maxage:` segment, and a source
    /// whose first segment the server reads ahead of the source,
    /// signed or not, keyed or not, plain or sealed, with the
    /// one typed error and the one message; without the expiry
    /// the same inputs build as they always did. The vectors
    /// that build mark where the refusal stops.
    #[test]
    fn expiry_test_vectors() {
        let vectors: Vec<serde_json::Value> = serde_json::from_str(VECTORS).unwrap();
        let base = "https://img.example.com";
        let secret = &b"s3cr3t"[..];
        let (mut refusals, mut builds) = (0, 0);
        for v in &vectors {
            let description = v["description"].as_str().unwrap();
            if let Some(refusal) = v.get("refuses_with_expiry") {
                refusals += 1;
                let operations = v["operations"].as_str().unwrap();
                let source = v["source"].as_str().unwrap();
                let expiry = v["expires"].as_u64().unwrap();
                let expires = Some(expiry);
                let what = match refusal["input"].as_str().unwrap() {
                    "operations" => "operations",
                    "source" => "source",
                    other => panic!("{description}: no input {other}"),
                };
                let segment = refusal["segment"].as_str().unwrap();
                let refused = BuildError::ExpiryOverride {
                    what,
                    segment: segment.to_string(),
                };
                assert_eq!(refused.to_string(), expiry_override_message(what, segment));
                let plain = if operations.is_empty() {
                    source.to_string()
                } else {
                    format!("{operations}/{source}")
                };
                for secret in [Some(secret), None] {
                    assert_eq!(
                        build_url(base, secret, operations, source, expires),
                        Err(refused.clone()),
                        "{description}"
                    );
                    let url = build_url(base, secret, operations, source, None).unwrap();
                    assert!(url.ends_with(&format!("/{plain}")), "{description}: {url}");
                }
                assert_eq!(
                    build_url_with_key_id(base, (secret, "k2"), operations, source, expires),
                    Err(refused.clone()),
                    "{description}"
                );
                if what == "source" {
                    assert_eq!(
                        build_srcset(base, Some(secret), source, &[320], expires),
                        Err(refused.clone()),
                        "{description}"
                    );
                }
                let sealed = SealError::ExpiryOverride {
                    what,
                    segment: segment.to_string(),
                };
                assert_eq!(sealed.to_string(), refused.to_string());
                let url = SealedUrl::new(base, operations, source).expires(expiry);
                for url in [url, url.key_id("k2"), url.name("a.jpg")] {
                    assert_eq!(url.build(secret), Err(sealed.clone()), "{description}");
                }
                assert!(
                    SealedUrl::new(base, operations, source)
                        .build(secret)
                        .is_ok()
                );
                for secret in [Some(secret), None] {
                    #[allow(deprecated)]
                    let shim =
                        build_sealed_url(base, secret, operations, source, expires, None, None);
                    assert_eq!(shim, Err(sealed.clone()), "{description}");
                }
            }
            if let Some(remainder) = v.get("builds_with_expiry") {
                builds += 1;
                let operations = v["operations"].as_str().unwrap();
                let source = v["source"].as_str().unwrap();
                let expiry = v["expires"].as_u64().unwrap();
                let expires = Some(expiry);
                let remainder = remainder.as_str().unwrap();
                assert_eq!(
                    build_url(base, Some(secret), operations, source, expires).unwrap(),
                    format!("{base}/{}/{remainder}", sign(secret, remainder).unwrap()),
                    "{description}"
                );
                assert_eq!(
                    build_url(base, None, operations, source, expires).unwrap(),
                    format!("{base}/unsafe/{remainder}"),
                    "{description}"
                );
                let url = SealedUrl::new(base, operations, source)
                    .expires(expiry)
                    .build(secret)
                    .unwrap();
                let token = url.strip_prefix(&format!("{base}/")).unwrap();
                assert_eq!(unseal(secret, token).unwrap(), remainder, "{description}");
            }
        }
        assert!(refusals >= 11, "the spec carries expiry refusal vectors");
        assert!(builds >= 5, "the spec carries expiry build vectors");
    }

    /// The segments the expiry check reads ahead of the source
    /// (issue #782): each prefix the server's parser reads as
    /// metadata, a preset reference or an operation, and a size
    /// in any spelling the parser reads as one, valid or not.
    /// The server's tests hold the check to its parser over the
    /// grammar corpora.
    #[test]
    fn the_expiry_check_reads_segments_as_the_server_does() {
        for segment in [
            "exp:1",
            "maxage:60",
            "preset:og-card",
            "debug:focal",
            "frame:poster",
            "page:2",
            "crop-rect:0x0:10x10",
            "focalpoint:0.5,0.5,2",
            "focal:0x0:10x10",
            "crop:100x100:smart",
            "pad:100x100",
            "fit-in:100x100",
            "filters:grayscale",
            "300x200",
            "-300x0",
            "origx100",
            "x",
            "16384x10",
            "300xabc",
            "xmas",
        ] {
            assert!(read_before_source(segment), "{segment}");
        }
        for segment in [
            "",
            "https:",
            "s3:",
            "gs:",
            "file:",
            "root:photos",
            "example.com",
            "100x200.jpg",
            "x.jpg",
            "photo.jpg",
            "crabcrop",
            "trim",
            "filters",
            "300",
        ] {
            assert!(!read_before_source(segment), "{segment}");
        }
    }

    /// Each derived subkey is the HMAC of its label under the
    /// secret, so a signature over a label would be that
    /// subkey: no signer in this crate computes one, and no
    /// builder assembles a label from operations and a source,
    /// with or without an expiry or a key ID.
    #[test]
    fn no_signer_returns_a_derived_subkey() {
        let secret = b"label-test-secret-0123456789";
        let base = "https://img.example.com";
        type Derive = fn(&[u8]) -> Vec<u8>;
        let subkeys: [(&str, Derive); 5] = [
            ("crabcrop/purge-key", purge_key),
            ("crabcrop/purge-group-key", purge_group_key),
            ("crabcrop/upload-key", upload_key),
            ("crabcrop/seal-key", seal_key),
            (ENTRY_KEY_LABEL, |secret| {
                derive_subkey(secret, ENTRY_KEY_LABEL)
            }),
        ];
        for (label, subkey) in subkeys {
            // The derivation is unchanged: the HMAC of the
            // label under the secret, as the spec states.
            let key = subkey(secret);
            assert_eq!(key, hmac_sha256(secret, label.as_bytes()).to_vec());
            let refused = Err(BuildError::ReservedRemainder(label.to_string()));
            assert_eq!(sign(secret, label), refused, "{label}");
            assert_eq!(build_url(base, Some(secret), "", label, None), refused);
            let (head, tail) = label.split_once('/').unwrap();
            assert_eq!(build_url(base, Some(secret), head, tail, None), refused);
            assert!(build_url_with_key_id(base, (secret, "k"), "", label, None).is_err());
            assert!(SealedUrl::new(base, "", label).build(secret).is_err());
            // What does sign is never a subkey.
            for remainder in [
                format!("exp:1/{label}"),
                format!("300x200/{label}"),
                format!("/{label}"),
                label.to_uppercase(),
            ] {
                let signature = sign(secret, &remainder).unwrap();
                for (_, other) in subkeys {
                    assert_ne!(hex::decode(&signature).unwrap(), other(secret));
                }
            }
        }
    }

    #[test]
    fn reserved_remainders_are_the_reserved_segment_and_below() {
        for reserved in ["crabcrop", "crabcrop/", "crabcrop/x", "crabcrop//x"] {
            assert!(is_reserved_remainder(reserved), "{reserved}");
        }
        for open in [
            "",
            "crabcro",
            "crabcropx",
            "crabcrop-assets/photo.jpg",
            "Crabcrop/purge-key",
            "/crabcrop/purge-key",
            "exp:1/crabcrop/purge-key",
            "300x200/https://example.com/crabcrop/purge-key",
        ] {
            assert!(!is_reserved_remainder(open), "{open}");
        }
    }

    /// The fix the refusal message names works: the
    /// percent-encoded spelling builds, signs as written (the
    /// spec vector for this remainder), and contains nothing
    /// a browser would rewrite.
    #[test]
    fn percent_encoded_sources_build_and_sign_as_written() {
        let url = build_url(
            "https://img.example.com",
            Some(b"s3cr3t"),
            "100x0",
            "https://example.com/x%7By%7D.jpg",
            None,
        )
        .unwrap();
        assert_eq!(
            url,
            "https://img.example.com/185ec1ffdc665285c43ca7af231f4bef87fe8c6f102709006e46e8ace401da57/100x0/https://example.com/x%7By%7D.jpg"
        );
        let err = build_url(
            "https://img.example.com",
            Some(b"s3cr3t"),
            "100x0",
            "https://example.com/x{y}.jpg",
            None,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "source contains a character that cannot appear in a crabcrop URL: '{' \
             (browsers rewrite it; percent-encode it as %7B)"
        );
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
        assert_ne!(key, derive_subkey(secret, ENTRY_KEY_LABEL));
        assert_ne!(key, upload_key(secret));
        assert_ne!(key, secret.to_vec());
        assert_eq!(key, seal_key(secret));
        // The assembled URL: token in the signature slot,
        // the friendly name after it, nothing else.
        let url = SealedUrl::new(
            "https://img.example.com/",
            "300x200",
            "https://example.com/photo.jpg",
        )
        .expires(1893456000)
        .name("photo.jpg")
        .build(secret)
        .unwrap();
        let path = url.strip_prefix("https://img.example.com/").unwrap();
        let (segment, name) = path.split_once('/').unwrap();
        assert_eq!(name, "photo.jpg");
        assert_eq!(
            unseal(secret, segment).unwrap(),
            "exp:1893456000/300x200/https://example.com/photo.jpg"
        );
        // No name, or an empty one: the URL ends at the token.
        for url in [
            SealedUrl::new(
                "https://img.example.com",
                "300x200",
                "https://example.com/photo.jpg",
            ),
            SealedUrl::new(
                "https://img.example.com",
                "300x200",
                "https://example.com/photo.jpg",
            )
            .name(""),
        ] {
            let bare = url.key_id("k2").build(secret).unwrap();
            let segment = bare.strip_prefix("https://img.example.com/").unwrap();
            assert!(segment.starts_with("k2:s1."), "{segment}");
            assert_eq!(
                unseal(secret, segment).unwrap(),
                "300x200/https://example.com/photo.jpg"
            );
        }
        // Unbuildable remainders refuse at build time, like
        // the plain builders (audit P-4), and so does a bad
        // key ID.
        let unbuildable = SealedUrl::new(
            "https://img.example.com",
            "300x200",
            "https://example.com/a.jpg?w=1",
        );
        assert_eq!(
            unbuildable.build(secret),
            Err(SealError::InvalidCharacter {
                what: "source",
                character: '?'
            })
        );
        let url = SealedUrl::new("https://img.example.com", "300x200", "a.jpg");
        assert_eq!(
            url.key_id("has space").build(secret),
            Err(SealError::InvalidKeyId("has space".to_string()))
        );
        // The sealing cap applies to the assembled remainder.
        let long = "a".repeat(MAX_SEALED_REMAINDER);
        let at_cap = SealedUrl::new("https://img.example.com", "", &long)
            .build(secret)
            .unwrap();
        assert_eq!(unseal(secret, &unseal_shape(&at_cap)).unwrap(), long);
        assert_eq!(
            SealedUrl::new("https://img.example.com", "x", &long).build(secret),
            Err(SealError::RemainderTooLong(MAX_SEALED_REMAINDER + 2))
        );
    }

    /// The segment after the base of a sealed URL with no
    /// friendly name: the token itself.
    fn unseal_shape(url: &str) -> String {
        url.strip_prefix("https://img.example.com/")
            .unwrap()
            .to_string()
    }

    /// The deprecated positional builder (issue #712) answers
    /// what [`SealedUrl`] answers for the same inputs, and a
    /// `None` secret is [`SealError::NoSecret`] once the inputs
    /// pass their checks.
    #[test]
    #[allow(deprecated)]
    fn the_positional_sealed_builder_is_the_struct_builder() {
        let base = "https://img.example.com";
        let secret = b"shim-secret";
        assert_eq!(
            build_sealed_url(base, None, "300x200", "a.jpg", None, None, None),
            Err(SealError::NoSecret)
        );
        assert_eq!(
            build_sealed_url(base, None, "300x200", "a.jpg?", None, None, None),
            Err(SealError::InvalidCharacter {
                what: "source",
                character: '?'
            })
        );
        let url = build_sealed_url(
            base,
            Some(secret),
            "300x200",
            "a.jpg",
            Some(7),
            Some("n.jpg"),
            Some("k1"),
        )
        .unwrap();
        let (segment, name) = unseal_shape(&url)
            .split_once('/')
            .map(|(s, n)| (s.to_string(), n.to_string()))
            .unwrap();
        assert_eq!(name, "n.jpg");
        assert!(segment.starts_with("k1:s1."), "{segment}");
        assert_eq!(unseal(secret, &segment).unwrap(), "exp:7/300x200/a.jpg");
        // Tokens carry a fresh nonce, so compare the shape and
        // the payload, not the bytes.
        let built = SealedUrl::new(base, "300x200", "a.jpg")
            .expires(7)
            .name("n.jpg")
            .key_id("k1")
            .build(secret)
            .unwrap();
        let (built_segment, built_name) = unseal_shape(&built)
            .split_once('/')
            .map(|(s, n)| (s.to_string(), n.to_string()))
            .unwrap();
        assert_eq!(built_name, name);
        assert_eq!(
            unseal(secret, &built_segment).unwrap(),
            unseal(secret, &segment).unwrap()
        );
    }

    /// Sealing fails with its own errors, never with
    /// [`SealError::OpenFailed`], which belongs to opening
    /// (issue #712), and each one says what happened.
    #[test]
    fn seal_failures_are_not_open_failures() {
        let messages = [
            (
                SealError::RandomnessFailed("no entropy source".to_string()),
                "the operating system's random number generator failed, so no nonce could be \
                 drawn: no entropy source",
            ),
            (
                SealError::EncryptFailed,
                "AES-256-GCM refused to encrypt the remainder",
            ),
            (
                SealError::OpenFailed,
                "the token did not open under this secret",
            ),
        ];
        for (error, message) in messages {
            assert_eq!(error.to_string(), message);
        }
        assert_ne!(SealError::EncryptFailed, SealError::OpenFailed);
        assert_ne!(
            SealError::RandomnessFailed(String::new()),
            SealError::OpenFailed
        );
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
            sign(b"test-secret", "300x200/https://example.com/a.jpg").unwrap()
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
    fn expiring_purge_signatures_cover_the_expiry() {
        // Pinned literals, computed with Python's hmac module
        // and with OpenSSL: the expiring form signs
        // `exp:<expires>/` and the target under each
        // endpoint's own subkey.
        let secret = b"test-secret";
        let target = "300x200/https://example.com/a.jpg";
        let exp = 1893456000;
        assert_eq!(purge_signed_string(target, None), target);
        assert_eq!(
            purge_signed_string("thumb", Some(exp)),
            "exp:1893456000/thumb"
        );
        assert_eq!(
            purge_signature_with_expiry(secret, target, exp),
            "fb745f8bc540df658be4c4c954b49054e871bd7c6836be43edb74a34a01aacb0"
        );
        assert_eq!(
            purge_group_signature_with_expiry(secret, "thumb", exp),
            "fd5eb666f654c3046374cd6f569a23f9a0254be1f1761ee16255c217a2b40a3f"
        );
        // Bound to the expiry, the target and the endpoint,
        // and never the plain signature.
        assert_ne!(
            purge_signature_with_expiry(secret, target, exp),
            purge_signature_with_expiry(secret, target, exp + 1)
        );
        assert_ne!(
            purge_signature_with_expiry(secret, target, exp),
            purge_signature(secret, target)
        );
        assert_ne!(
            purge_group_signature_with_expiry(secret, "thumb", exp),
            purge_group_signature(secret, "thumb")
        );
        assert_ne!(
            purge_group_signature_with_expiry(secret, "thumb", exp),
            purge_signature_with_expiry(secret, "thumb", exp)
        );
        // A plain target that opens with an image URL's exp:
        // segment signs the same bytes as the expiring form
        // of the rest, which is why the server reads that
        // segment as the purge's own expiry.
        assert_eq!(
            purge_signature(secret, &format!("exp:{exp}/{target}")),
            purge_signature_with_expiry(secret, target, exp)
        );
    }

    #[test]
    fn derived_subkeys_are_domain_separated() {
        // Purge, purge-group, and upload: same secret, three
        // unrelated keys, none of them the secret itself. The
        // server's entry key is a fourth, which its own tests
        // check beside these.
        let secret = b"test-secret";
        let keys = [
            purge_key(secret),
            purge_group_key(secret),
            upload_key(secret),
        ];
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(key.len(), 32);
            assert_ne!(*key, secret.to_vec());
            for other in &keys[i + 1..] {
                assert_ne!(key, other);
            }
        }
        // And each is deterministic: two derivations agree,
        // so every replica holding the secret verifies what
        // any other signed.
        assert_eq!(keys[0], purge_key(secret));
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
            sign(b"test-secret", "heroes/cat.jpg").unwrap()
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
            format!(
                "k2:{}",
                sign(b"test-secret", "300x200/example.com/i.jpg").unwrap()
            )
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
            (b"test-secret", "k2"),
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
        // A bad key ID is the one refusal of its own.
        assert_eq!(
            build_url_with_key_id(
                "https://img.example.com",
                (b"test-secret", "bad id"),
                "300x200",
                "example.com/i.jpg",
                None,
            ),
            Err(BuildError::InvalidKeyId("bad id".to_string()))
        );
    }
}
