//! Property tests for signing and sealing (issue #478): for
//! arbitrary secrets and remainders up to
//! [`MAX_SEALED_REMAINDER`], a signature verifies and a
//! sealed token opens, keyed and bare, and changing any one
//! byte of the remainder, of the decoded tag, or of the
//! decoded token makes them fail. `hex::decode` accepts
//! either case, so the tag is mutated as bytes, never as hex
//! digits (a case flip is the same tag).
//!
//! The cases per property default to [`DEFAULT_CASES`];
//! `PROPTEST_CASES` raises it (the nightly job does). A
//! failure is saved next to this file in
//! `proptest_sign.proptest-regressions`, which is checked in
//! so the seed replays first on every later run.

use base64::Engine as _;
use crabcrop_url::{
    MAX_SEALED_REMAINDER, SEALED_PREFIX, SealError, build_sealed_url, build_url,
    build_url_with_key_id, open_token, seal, seal_with_key_id, seal_with_nonce, sign,
    sign_with_key_id, unseal, verify,
};
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::FileFailurePersistence;

/// Cases per property when `PROPTEST_CASES` is unset: enough
/// to walk every branch on each `cargo test`, few enough to
/// keep the suite to a second or so in a debug build.
const DEFAULT_CASES: u32 = 64;

fn config() -> ProptestConfig {
    let mut config = ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource(
            "proptest-regressions",
        ))),
        ..ProptestConfig::default()
    };
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = DEFAULT_CASES;
    }
    config
}

/// A secret: arbitrary bytes, empty and NUL included (the
/// API takes bytes, and a secret drawn from a random source
/// holds NULs).
fn secret() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=96)
}

/// Whether two secrets may be one key. HMAC zero-pads a key
/// shorter than its block (RFC 2104), so secrets that differ
/// only by trailing NULs sign and seal alike; "another
/// secret" must differ by more than that.
fn may_be_one_key(a: &[u8], b: &[u8]) -> bool {
    let trim = |s: &[u8]| s[..s.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1)].to_vec();
    trim(a) == trim(b)
}

/// `secret` with the byte at `at` changed: always another
/// key, since the change survives the zero padding and a key
/// longer than the block is hashed whole.
fn change_secret_byte(secret: &[u8], at: usize, flip: u8) -> Vec<u8> {
    let mut changed = secret.to_vec();
    changed[at] ^= flip;
    changed
}

/// A key ID per the grammar: 1 to 32 of `[A-Za-z0-9_-]`.
fn key_id() -> impl Strategy<Value = String> {
    "[A-Za-z0-9_-]{1,32}"
}

/// Any text within the sealing cap: arbitrary characters,
/// so multi-byte UTF-8 and control characters are covered
/// (the primitives sign and seal bytes; only the builders
/// police characters).
fn any_remainder() -> impl Strategy<Value = String> {
    prop_oneof![
        prop::collection::vec(any::<char>(), 0..=MAX_SEALED_REMAINDER / 4)
            .prop_map(String::from_iter),
        ascii_remainder(),
    ]
}

/// Printable ASCII within the sealing cap, the cap itself
/// weighted in, so a byte can be changed in place and the
/// text stays a `&str`.
fn ascii_remainder() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => prop::collection::vec(0x20u8..0x7f, 0..=MAX_SEALED_REMAINDER),
        1 => prop::collection::vec(0x20u8..0x7f, MAX_SEALED_REMAINDER..=MAX_SEALED_REMAINDER),
    ]
    .prop_map(|bytes| String::from_utf8(bytes).expect("printable ASCII"))
}

/// Builder input that travels unchanged in a URL path under
/// both the current builders and the stricter set of issue
/// #472 (which also refuses `"` `<` `>` `\` `^` `` ` `` `{`
/// `|` `}`): unreserved characters, sub-delimiters, `:`,
/// `@`, `%`, and `/`.
const URL_SAFE: &str = "[A-Za-z0-9._~!$&'()*+,;=:@%/-]";

/// Builder input of `min` to `max` [`URL_SAFE`] characters.
fn url_text(min: usize, max: usize) -> impl Strategy<Value = String> {
    proptest::string::string_regex(&format!("{URL_SAFE}{{{min},{max}}}")).expect("a valid regex")
}

/// Change the byte at `at` to a different printable ASCII
/// byte: `delta` in 1..95 steps around the 95 printable
/// characters, so the result is never the original.
fn change_ascii_byte(text: &str, at: usize, delta: u8) -> String {
    let mut bytes = text.as_bytes().to_vec();
    bytes[at] = (bytes[at] - 0x20 + delta) % 95 + 0x20;
    String::from_utf8(bytes).expect("printable ASCII")
}

/// The token body's bytes: nonce, ciphertext, and tag.
fn token_bytes(token: &str) -> Vec<u8> {
    let body = token.strip_prefix(SEALED_PREFIX).expect("a sealed token");
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(body)
        .expect("a minted token is base64url")
}

fn token_from(bytes: &[u8]) -> String {
    format!(
        "{SEALED_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    )
}

proptest! {
    #![proptest_config(config())]

    /// A signature is 64 lowercase hex characters and
    /// verifies over the remainder it was minted for, bare
    /// and behind a key ID.
    #[test]
    fn signatures_verify_keyed_and_bare(
        secret in secret(),
        remainder in any_remainder(),
        kid in key_id(),
    ) {
        let tag = sign(&secret, &remainder);
        prop_assert_eq!(tag.len(), 64);
        prop_assert!(tag.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        prop_assert!(verify(&secret, &tag, &remainder));
        let keyed = sign_with_key_id(&secret, &kid, &remainder).unwrap();
        prop_assert_eq!(keyed, format!("{kid}:{tag}"));
    }

    /// Changing any one byte of the remainder or of the
    /// decoded tag fails verification, as do a tag one byte
    /// short or long and a different secret. Each case
    /// changes the first byte, the last, and one at random,
    /// so a bug that skips an end of the input fails every
    /// run, not one in a few hundred cases.
    #[test]
    fn signatures_refuse_one_byte_changes(
        secret in secret(),
        other in secret(),
        remainder in ascii_remainder().prop_filter("a byte to change", |r| !r.is_empty()),
        at in any::<Index>(),
        delta in 1u8..95,
        tag_at in 0usize..32,
        secret_at in any::<Index>(),
        flip in 1u8..=255,
    ) {
        let tag = sign(&secret, &remainder);
        let len = remainder.len();
        for at in [0, len - 1, at.index(len)] {
            let changed = change_ascii_byte(&remainder, at, delta);
            prop_assert!(
                !verify(&secret, &tag, &changed),
                "remainder byte {} changed",
                at
            );
        }

        let mut raw = hex::decode(&tag).unwrap();
        for at in [0, 31, tag_at] {
            raw[at] ^= flip;
            prop_assert!(
                !verify(&secret, &hex::encode(&raw), &remainder),
                "tag byte {} changed",
                at
            );
            raw[at] ^= flip;
        }
        prop_assert!(!verify(&secret, &hex::encode(&raw[..31]), &remainder));
        raw.push(0);
        prop_assert!(!verify(&secret, &hex::encode(&raw), &remainder));

        if !secret.is_empty() {
            let changed = change_secret_byte(&secret, secret_at.index(secret.len()), flip);
            prop_assert!(!verify(&changed, &tag, &remainder));
        }
        prop_assume!(!may_be_one_key(&other, &secret));
        prop_assert!(!verify(&other, &tag, &remainder));
    }

    /// A sealed token opens to its remainder under the
    /// minting secret, keyed and bare, with a pinned or a
    /// random nonce, and never opens under another secret.
    #[test]
    fn sealed_tokens_open_keyed_and_bare(
        secret in secret(),
        other in secret(),
        remainder in any_remainder(),
        kid in key_id(),
        nonce in any::<[u8; 12]>(),
        secret_at in any::<Index>(),
        flip in 1u8..=255,
    ) {
        let token = seal_with_nonce(&secret, None, &nonce, &remainder).unwrap();
        prop_assert!(token.starts_with(SEALED_PREFIX));
        prop_assert_eq!(unseal(&secret, &token), Ok(remainder.clone()));
        prop_assert_eq!(open_token(&secret, &token), Ok(remainder.clone()));

        let keyed = seal_with_nonce(&secret, Some(&kid), &nonce, &remainder).unwrap();
        prop_assert_eq!(&keyed, &format!("{kid}:{token}"));
        prop_assert_eq!(unseal(&secret, &keyed), Ok(remainder.clone()));

        let fresh = seal(&secret, &remainder).unwrap();
        prop_assert_eq!(unseal(&secret, &fresh), Ok(remainder.clone()));
        let fresh_keyed = seal_with_key_id(&secret, &kid, &remainder).unwrap();
        prop_assert_eq!(unseal(&secret, &fresh_keyed), Ok(remainder.clone()));

        if !secret.is_empty() {
            let changed = change_secret_byte(&secret, secret_at.index(secret.len()), flip);
            prop_assert_eq!(unseal(&changed, &token), Err(SealError::OpenFailed));
        }
        prop_assume!(!may_be_one_key(&other, &secret));
        prop_assert_eq!(unseal(&other, &token), Err(SealError::OpenFailed));
        prop_assert_eq!(unseal(&other, &keyed), Err(SealError::OpenFailed));
    }

    /// Changing any one byte of the decoded token (nonce,
    /// ciphertext, or tag) or cutting it short fails the open
    /// with the one opaque error, keyed and bare. Each case
    /// changes the first byte (the nonce's), the last (the
    /// tag's), and one at random.
    #[test]
    fn sealed_tokens_refuse_one_byte_changes(
        secret in secret(),
        remainder in any_remainder(),
        kid in key_id(),
        nonce in any::<[u8; 12]>(),
        at in any::<Index>(),
        flip in 1u8..=255,
        cut in any::<Index>(),
    ) {
        let token = seal_with_nonce(&secret, None, &nonce, &remainder).unwrap();
        let mut raw = token_bytes(&token);
        let len = raw.len();
        prop_assert_eq!(len, 12 + remainder.len() + 16);
        for at in [0, len - 1, at.index(len)] {
            raw[at] ^= flip;
            let damaged = token_from(&raw);
            prop_assert_eq!(
                unseal(&secret, &damaged),
                Err(SealError::OpenFailed),
                "token byte {} changed",
                at
            );
            prop_assert_eq!(
                unseal(&secret, &format!("{kid}:{damaged}")),
                Err(SealError::OpenFailed)
            );
            raw[at] ^= flip;
        }
        let short = token_from(&raw[..cut.index(len)]);
        prop_assert_eq!(unseal(&secret, &short), Err(SealError::OpenFailed));
    }

    /// Sealing refuses a remainder over the cap with its
    /// length, keyed and bare, and mints nothing.
    #[test]
    fn sealing_refuses_remainders_over_the_cap(
        secret in secret(),
        len in MAX_SEALED_REMAINDER + 1..=2 * MAX_SEALED_REMAINDER,
        kid in key_id(),
    ) {
        let remainder = "a".repeat(len);
        prop_assert_eq!(seal(&secret, &remainder), Err(SealError::RemainderTooLong(len)));
        prop_assert_eq!(
            seal_with_key_id(&secret, &kid, &remainder),
            Err(SealError::RemainderTooLong(len))
        );
    }

    /// The builders assemble the remainder the way the spec
    /// says and sign or seal exactly it: the signature in a
    /// built URL verifies over the rest of the path, a keyed
    /// one carries its ID, and a sealed URL's token opens to
    /// the same remainder. The inputs use only characters
    /// both the current builders and the issue #472 builders
    /// accept, so the property holds across that change.
    #[test]
    fn built_urls_verify_and_open(
        secret in secret(),
        operations in url_text(0, 40),
        source in url_text(1, 60),
        expires in proptest::option::of(any::<u64>()),
        kid in key_id(),
        slashes in 0usize..3,
    ) {
        let base = format!("https://img.example.com{}", "/".repeat(slashes));
        let mut remainder = String::new();
        if let Some(exp) = expires {
            remainder.push_str(&format!("exp:{exp}/"));
        }
        if !operations.is_empty() {
            remainder.push_str(&operations);
            remainder.push('/');
        }
        remainder.push_str(&source);

        let url = build_url(&base, Some(&secret), &operations, &source, expires).unwrap();
        let path = url.strip_prefix("https://img.example.com/").unwrap();
        let (tag, rest) = path.split_once('/').unwrap();
        prop_assert_eq!(rest, &remainder);
        prop_assert!(verify(&secret, tag, rest));

        let unsigned = build_url(&base, None, &operations, &source, expires).unwrap();
        prop_assert_eq!(unsigned, format!("https://img.example.com/unsafe/{remainder}"));

        let keyed =
            build_url_with_key_id(&base, Some((&secret, &kid)), &operations, &source, expires)
                .unwrap();
        prop_assert_eq!(keyed, format!("https://img.example.com/{kid}:{tag}/{remainder}"));

        let sealed =
            build_sealed_url(&base, Some(&secret), &operations, &source, expires, None, Some(&kid))
                .unwrap();
        let segment = sealed.strip_prefix("https://img.example.com/").unwrap();
        prop_assert_eq!(unseal(&secret, segment), Ok(remainder.clone()));
    }
}
