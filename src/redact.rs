//! What a secret prints as in `Debug` output (issue #455).
//!
//! The types that hold key material, credentials, or tokens
//! (the signing keyring, bucket credentials, the stores'
//! entry MAC key, the `/metrics` token) implement `Debug` by
//! hand and print each secret field as [`Redacted`]. A `{:?}`,
//! a `dbg!`, or a panic message that formats one of them then
//! shows that the secret is set, never its bytes. Types that
//! only contain those types keep a derived `Debug`, so the
//! redaction holds wherever they are nested.

/// Prints as `[redacted]`.
pub(crate) struct Redacted;

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[redacted]")
    }
}
