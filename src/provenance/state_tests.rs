//! The validation state meta reports (issue #564). The
//! reader checks signatures, not trust lists: it loads no
//! trust anchors, so a manifest reads `valid` or
//! `invalid`, and never `trusted`, whoever issued its
//! certificate.

use super::*;

/// The test CA's certificate: the second one in the test
/// chain.
fn test_ca_pem() -> &'static str {
    let (start, _) = TEST_CHAIN
        .match_indices("-----BEGIN CERTIFICATE-----")
        .nth(1)
        .expect("the test chain carries its CA");
    &TEST_CHAIN[start..]
}

/// A reader over `signed` whose context trusts `anchors`.
fn reader_trusting(signed: &[u8], anchors: &str) -> Reader {
    let settings = serde_json::json!({ "trust": { "trust_anchors": anchors } }).to_string();
    let context = Context::new()
        .with_settings(settings.as_str())
        .expect("the trust settings load");
    Reader::from_context(context)
        .with_stream("image/png", Cursor::new(signed))
        .expect("the reader parses the output")
}

#[test]
fn a_chain_under_a_trust_anchor_still_reads_valid() {
    let signed = test_signer()
        .sign(
            SELF_TEST_PNG,
            "image/png",
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .expect("signing works");
    // With the test CA as an anchor, the SDK itself calls
    // the chain trusted: this is the state a reader with
    // trust anchors would reach.
    let trusting = reader_trusting(&signed, test_ca_pem());
    assert_eq!(trusting.validation_state(), ValidationState::Trusted);
    // Meta makes no trust claim, whatever the SDK says.
    assert_eq!(summarize(&trusting).state, "valid");
    // And meta's own read, with no anchors, says valid.
    let summary = read_manifest(&signed, "image/png").expect("the output carries a manifest");
    assert_eq!(summary.state, "valid");
}
