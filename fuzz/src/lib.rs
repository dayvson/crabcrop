//! The fuzz harness for the URL parser (issue #146).
//!
//! The parser lives in the server binary crate, which a fuzz
//! package cannot depend on, so the module is included by
//! path: it has no crate-internal dependencies (only the
//! `crabcrop-url` signing crate and std), and compiles here
//! unchanged. The same `check` drives the cargo-fuzz target
//! and the deterministic smoke test.

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
        ("media".to_string(), "https://assets.example.com".to_string()),
    ]
    .into_iter()
    .collect();
    let _ = url::expand_source_root(input, &roots);
    let _ = url::expand_source_root_in_remainder(input, &roots);
    if let Ok((_, _, request)) = url::parse_request(input) {
        if let Ok((expanded, Some(reference))) =
            url::expand_source_root(request.source_url, &roots)
        {
            let base = &roots[&reference.name];
            assert!(
                expanded.starts_with(&format!("{base}/")),
                "root expansion escaped its base: {expanded}"
            );
        }
    }
}
