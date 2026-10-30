//! The fuzz harness for the two untrusted-text surfaces in
//! front of the server: the URL parser (issue #146) and the
//! sealed-token opener (issue #255).
//!
//! The parser lives in the server binary crate, which a fuzz
//! package cannot depend on, so the module is included by
//! path: it has no crate-internal dependencies (only the
//! `crabcrop-url` signing crate and std), and compiles here
//! unchanged. The opener lives in `crabcrop-url` itself, a
//! direct dependency. `check` drives the parser target and
//! `check_seal` the opener target; both also run under the
//! deterministic smoke test.

#[path = "../../src/url.rs"]
pub mod url;

use url::Signature;

/// The round-trip invariants every input must satisfy,
/// however malformed:
///
/// - Nothing panics: parsing, canonicalizing, and
///   re-serializing untrusted text is total.
/// - A path that parses canonicalizes to a fixed point:
///   canonical form, re-parsed and re-canonicalized, is
///   byte-identical.
/// - The canonical key form is likewise a fixed point.
pub fn check(input: &str) {
    if let Ok((signature, key_id, request)) = url::parse_request(input) {
        let canonical = url::canonical_form(&request.operations, request.source_url);
        let segment = match &signature {
            Signature::Unsafe => "unsafe".to_string(),
            Signature::Hmac(hex) => match &key_id {
                Some(kid) => format!("{kid}:{hex}"),
                None => hex.clone(),
            },
        };
        let rebuilt = format!("/{segment}/{canonical}");
        let reparsed = url::parse_request(&rebuilt);
        let (sig2, kid2, request2) = reparsed.expect("the canonical form must re-parse");
        assert_eq!(sig2, signature, "signature changed across canonicalization");
        assert_eq!(kid2, key_id, "key ID changed across canonicalization");
        let canonical2 = url::canonical_form(&request2.operations, request2.source_url);
        assert_eq!(canonical, canonical2, "canonical form is not a fixed point");
    }
    if let Ok((_, _, rest)) = url::parse_signature(input) {
        let key1 = url::canonical_key_form(rest);
        let key2 = url::canonical_key_form(&key1);
        assert_eq!(key1, key2, "canonical key form is not a fixed point");
    }
    // The remaining entry points over untrusted text: total
    // by assertion (a panic fails the run).
    let _ = url::parse_signature(input);
    let _ = url::expand_presets(input, &std::collections::HashMap::new(), false);
    let _ = url::preset_names(input);
    // Root expansion (issue #250): total over untrusted
    // text, and a successful expansion stays under the
    // named root's base by construction. The map is a
    // fixed stand-in for operator configuration.
    let roots: std::collections::HashMap<String, String> = [
        (
            "photos".to_string(),
            "https://cdn.example.com/photos".to_string(),
        ),
        (
            "media".to_string(),
            "https://assets.example.com".to_string(),
        ),
    ]
    .into_iter()
    .collect();
    let _ = url::expand_source_root(input, &roots);
    let _ = url::expand_source_root_in_remainder(input, &roots);
    if let Ok((_, _, request)) = url::parse_request(input) {
        if let Ok((expanded, Some(reference))) = url::expand_source_root(request.source_url, &roots)
        {
            let base = &roots[&reference.name];
            assert!(
                expanded.starts_with(&format!("{base}/")),
                "root expansion escaped its base: {expanded}"
            );
        }
    }
}

/// The harness secret for the seal invariants, and a distinct
/// second secret the same token must never open under.
const SEAL_SECRET: &[u8] = b"crabcrop-fuzz-seal-secret-000001";
const SEAL_OTHER: &[u8] = b"crabcrop-fuzz-seal-secret-000002";

/// The seal-and-open invariants every input must satisfy
/// (issue #255): the opener is the second untrusted-text
/// surface, so opening attacker-controlled bytes is total,
/// and a token's confidentiality and integrity hold.
///
/// - Nothing panics: opening a token segment under a secret
///   is total, however malformed the bytes, keyed or bare.
/// - Round trip: a remainder within the cap seals and reopens
///   to itself under the minting secret, keyed and bare.
/// - Integrity: a token never opens to its remainder under a
///   different secret, so a wrong key or a tampered body is a
///   refusal, never another plaintext.
pub fn check_seal(data: &[u8]) {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    let _ = crabcrop_url::unseal(SEAL_SECRET, text);
    let _ = crabcrop_url::unseal(SEAL_OTHER, text);
    let _ = crabcrop_url::open_token(SEAL_SECRET, text);

    if text.len() <= crabcrop_url::MAX_SEALED_REMAINDER {
        if let Ok(token) = crabcrop_url::seal(SEAL_SECRET, text) {
            assert_eq!(
                crabcrop_url::unseal(SEAL_SECRET, &token).as_deref(),
                Ok(text),
                "a sealed token must reopen to its remainder",
            );
            assert_ne!(
                crabcrop_url::unseal(SEAL_OTHER, &token).as_deref(),
                Ok(text),
                "a token must not open to its remainder under a different secret",
            );
        }
        if let Ok(keyed) = crabcrop_url::seal_with_key_id(SEAL_SECRET, "fuzzkey", text) {
            assert_eq!(
                crabcrop_url::unseal(SEAL_SECRET, &keyed).as_deref(),
                Ok(text),
                "a keyed sealed token must reopen to its remainder",
            );
        }
    }
}

/// A token sealed under the harness secret with a fixed
/// nonce, for the smoke test's truncation and mutation sweeps
/// over a well-formed token. Deterministic, so a failing case
/// reproduces.
pub fn sample_token(remainder: &str) -> String {
    crabcrop_url::seal_with_nonce(SEAL_SECRET, None, &[0x5a; 12], remainder)
        .expect("a remainder within the cap seals")
}
