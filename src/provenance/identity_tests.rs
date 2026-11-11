//! Signer names and key types (issues #595 and #600), with
//! more test identities, generated offline like
//! [`TEST_CHAIN`] and valid until 2126. None signs anything
//! outside the suites.

use super::*;

/// The leaf common name of [`LONG_NAME_CHAIN`]: 174
/// characters over two lines, past the 64 RFC 5280 allows.
/// The certificate parser takes it as it is, so a source
/// signed with this identity names whatever its maker chose.
const LONG_NAME: &str = "crabcrop test signer with a long name\nIts second line is text the certificate's maker chose, and the whole name runs past the 64 characters RFC 5280 allows for a common name.";

/// An ES256 CA and end-entity certificate whose common name
/// is [`LONG_NAME`].
const LONG_NAME_CHAIN: &str = "-----BEGIN CERTIFICATE-----\nMIICgDCCAiegAwIBAgIUZojr6EaEyaUbv6k3dW+9sboz8QAwCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjExMTEwMDAwMDBaGA8yMTI2MTAxODAwMDAwMFowgdQxgbkwgbYG\nA1UEAwyBrmNyYWJjcm9wIHRlc3Qgc2lnbmVyIHdpdGggYSBsb25nIG5hbWUKSXRz\nIHNlY29uZCBsaW5lIGlzIHRleHQgdGhlIGNlcnRpZmljYXRlJ3MgbWFrZXIgY2hv\nc2UsIGFuZCB0aGUgd2hvbGUgbmFtZSBydW5zIHBhc3QgdGhlIDY0IGNoYXJhY3Rl\ncnMgUkZDIDUyODAgYWxsb3dzIGZvciBhIGNvbW1vbiBuYW1lLjEWMBQGA1UECgwN\nQ3JhYmNyb3AgVGVzdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABEMQg0g+q98H\n3hjMaG2YIWcST03+XvFR17R8G1waCmppJ6IITPQnsVj5SERi9G5c2pM2ZWy2ZOhu\n0mOT293u/b+jdTBzMAwGA1UdEwEB/wQCMAAwDgYDVR0PAQH/BAQDAgeAMBMGA1Ud\nJQQMMAoGCCsGAQUFBwMEMB0GA1UdDgQWBBQW85dKwaMwPZY9YVzMUi5CQTEC5jAf\nBgNVHSMEGDAWgBRKp7MjA2kh3QhLn/jYJQTuZOYe7DAKBggqhkjOPQQDAgNHADBE\nAiA5NWkzhxvuoBfP9AEfPR1+NlziwfQyH8rpLcsaJjm3uwIgLPIRiAzLYXi2yo/k\nHJ6d8ULuv9ANqDvjCcPyEH19At8=\n-----END CERTIFICATE-----\n-----BEGIN CERTIFICATE-----\nMIIBrDCCAVKgAwIBAgIUFVnPRHwbckksIA+XVerG26HTrRUwCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjExMTEwMDAwMDBaGA8yMTI2MTAxODAwMDAwMFowMzEZMBcGA1UE\nAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3AgVGVzdDBZMBMG\nByqGSM49AgEGCCqGSM49AwEHA0IABPlYm/hbzSV2unFWyTibqefcE/CBM2TBawJ4\nbvW1Eka8uQffaD+Z59SG+cicHtzbrM79csOCBRX8weq+KdavALGjQjBAMA8GA1Ud\nEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgEGMB0GA1UdDgQWBBRKp7MjA2kh3QhL\nn/jYJQTuZOYe7DAKBggqhkjOPQQDAgNIADBFAiBm9zwFyJgkw8LeOM+jt+i61NQT\nsqmkq7W++IH/zFoBIQIhANEAYoZZvyG/rhtrTbLP/XGFLUNeOs/zNbKynDj+2v/3\n-----END CERTIFICATE-----\n";
/// The private key for [`LONG_NAME_CHAIN`].
const LONG_NAME_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgLPTMmNo3ICXkD+Sm\ny8J9jsISWZitwbxbUzQXOlnS3XqhRANCAARDEINIPqvfB94YzGhtmCFnEk9N/l7x\nUde0fBtcGgpqaSeiCEz0J7FY+UhEYvRuXNqTNmVstmTobtJjk9vd7v2/\n-----END PRIVATE KEY-----\n";

/// An RSA-2048 CA and end-entity certificate ("crabcrop test
/// RSA signer"), in the C2PA certificate profile like the
/// others.
const RSA_CHAIN: &str = "-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIUDA16lH5Zadl4thnN9tH0jdCwNWAwDQYJKoZIhvcNAQEL\nBQAwNzEdMBsGA1UEAwwUY3JhYmNyb3AgdGVzdCBSU0EgQ0ExFjAUBgNVBAoMDUNy\nYWJjcm9wIFRlc3QwIBcNMjYxMTExMDAwMDAwWhgPMjEyNjEwMTgwMDAwMDBaMDsx\nITAfBgNVBAMMGGNyYWJjcm9wIHRlc3QgUlNBIHNpZ25lcjEWMBQGA1UECgwNQ3Jh\nYmNyb3AgVGVzdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALR110Pe\ncGWJhVZPYH3cIvQnwXRR+wVBousjsfLsimvgG4MnsMELj0YclaPeeIeWx/dnvqhD\nrTp1slqsd1B1B+QO5MiDUo32/SJkWg9cZXgL6YdY9NlEbrM/ifILtJFd8YCp7oxy\ngm03St1xqbr3kxCJhNLLoS4REtFRr7trMWorOgPgzfYVtJQ1oBnX7o7qYQr4tQwH\nuceMp6IlYG2j0DH1BOg7ttrHROrCw8tFvLHP1CLMAgbravIG232t863BvHF2TYwa\ny00NBsNZT93rU6PZt+TKh0o8KbAcRay2lZVRrYKPyiS1v65W2Yld890IPSq0ViC0\nwPb2NICn0dZ1rTMCAwEAAaN1MHMwDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8EBAMC\nB4AwEwYDVR0lBAwwCgYIKwYBBQUHAwQwHQYDVR0OBBYEFFxBqmogQvM/oavB5TnS\nGS27XORKMB8GA1UdIwQYMBaAFG7YBx/OJyp8rCZ+P1lqNVDv0MErMA0GCSqGSIb3\nDQEBCwUAA4IBAQAMvloyGzZirJp7vcgaMAEQjBrWnrV0Gfq5fx8CpiCpiECJ399G\n0dOUb83+WYcb8s42SOMxOaD8mChMmx6vj0yGvfgUG80S8T3OMeSaM7CbjUV7WZf5\ntjhfl6uFh25kaZCdbSd70/OSHK2VlSBr8SX5G0MJMBCXAQBMB1S/BWGxR4xg28V1\n5Beurut8xkf4dksjPxUcC9m2WBI3ISxJaKfLgKoq/oCSZTBb16OfUMV3kabV9FVg\nfA+C9ecFuXDGq4WzRBRB04WkjgMZThtaSUqLT3o8yA+AIkdW7O1lhlpEI3t4NMiS\n1zddftvP5GeWZBPL5UBhGHv9WKKVPxzUFtZJ\n-----END CERTIFICATE-----\n-----BEGIN CERTIFICATE-----\nMIIDQDCCAiigAwIBAgIUeZ/HyEkqpIfpKsQYRONYhwPdoB4wDQYJKoZIhvcNAQEL\nBQAwNzEdMBsGA1UEAwwUY3JhYmNyb3AgdGVzdCBSU0EgQ0ExFjAUBgNVBAoMDUNy\nYWJjcm9wIFRlc3QwIBcNMjYxMTExMDAwMDAwWhgPMjEyNjEwMTgwMDAwMDBaMDcx\nHTAbBgNVBAMMFGNyYWJjcm9wIHRlc3QgUlNBIENBMRYwFAYDVQQKDA1DcmFiY3Jv\ncCBUZXN0MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxfwp9MKd7qRe\nXO5qa6x1GC0xczwjnN8cwVatAwsA1ZNP0+EKG4W4t8dX3rMybo3xMU4YJAnXn0MD\nzwuP56qTT9eRMmriUX/+xiY3vpjXR6Pfw9ceH/I5XDlskeHrYF+jaSu/HV8TP1hs\nDSDsJMh3CRJn9wIePSKlQXXJvROOh6rid13/N0lIx68VH8CvGdXyr5Ru0TaHaK5J\nLC3nYYKppy5Td7pTMH088uCnn6floKv2sTVauGwiKKCQKJ+kak5i/OQJHc3B2wA3\n0tfVhjQBa8JE6eeSe3dOw1OJzQlNDhb1a9IcMJDxgRdb3YNIESQI/UmXXzQsJ8lO\nzkPRNw/wRwIDAQABo0IwQDAPBgNVHRMBAf8EBTADAQH/MA4GA1UdDwEB/wQEAwIB\nBjAdBgNVHQ4EFgQUbtgHH84nKnysJn4/WWo1UO/QwSswDQYJKoZIhvcNAQELBQAD\nggEBAEwlnSEel9Hdxa0j16HRs87dS2U5tUF/10Tqe5q9XNLIAtBQ5CR0/hmA39w+\nR++DD+mGCIPIurKoRff2V4F5YxAgklPFyM7uo/m7C53AIyglT5GhKtALd1oN3EYx\n8tfunLWvMjiLk08PTyTVHx0mIbJAV9ytWLc7ikYGl5CMkXZQmsbjHwiILN4tivRN\nyjOGPrwPo48ipNelzPrPJwi6BoLc/LIUgA1fcJB9eelVpniN4zw9QDCfvw4vrqPO\nwb/OSmExj5KKZsjqIRGcLJSXDS5ArBOL3FuoPmSd2uRbq47Ex7jgIzWn2FijOYoa\nsC+1MObw0G1Ihu/yl2bedTybTuc=\n-----END CERTIFICATE-----\n";
/// The private key for [`RSA_CHAIN`], in PKCS#8 form.
const RSA_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC0dddD3nBliYVW\nT2B93CL0J8F0UfsFQaLrI7Hy7Ipr4BuDJ7DBC49GHJWj3niHlsf3Z76oQ606dbJa\nrHdQdQfkDuTIg1KN9v0iZFoPXGV4C+mHWPTZRG6zP4nyC7SRXfGAqe6McoJtN0rd\ncam695MQiYTSy6EuERLRUa+7azFqKzoD4M32FbSUNaAZ1+6O6mEK+LUMB7nHjKei\nJWBto9Ax9QToO7bax0TqwsPLRbyxz9QizAIG62ryBtt9rfOtwbxxdk2MGstNDQbD\nWU/d61Oj2bfkyodKPCmwHEWstpWVUa2Cj8oktb+uVtmJXfPdCD0qtFYgtMD29jSA\np9HWda0zAgMBAAECggEABYqi2CIyCGQvNIXOfD7w0deQsL49u50W4Cqh8chCvepN\nXfsvidYNicUh8TzJLby7GkVUzPz7ClMH7ZHjPhTB78ZnTyL9O3/MqSw0uckIEqYm\n/zGDn1JjJdz7IvQqENa2LlHJkgWILMtoYKF3aVbl7LmUWxbGtc/vaH8eKt6xT0XE\nV5zGfTkFzBgSjoLinuDiF7g86cVDU7IjKIwIp6c4ucVaChRjh2dSGtuaqRZdHHyv\nBG6JaufU+6wK/v4uSHWGmJL86Ja1PHSrntS8OKaTGYuqeS7jUOe6yTZoSSBEuw1K\n4sLWWqk85wc57I5HstSevBNd24uz2gdEyVLlReKbZQKBgQDbuIn/YGpy/Eb5Dq/c\nSWho2G/6xxdIxXge5HIt+wEjyG+S+TCG3xrWFAMEQ+QBWG5Owp4u6JTdL8wsS7FJ\nJcFmD2chB8Q6McoQbP925RharXYpNCDoF1RK0/RdQadGasubSh3EHrAoy0B1G8qm\nUySiAx63sXXcAEe322fNJHweTwKBgQDSQcikeuQM+rk1Dql8U7DI3TN7vzlx1xbn\n7jRIz+R0METppHJ2zJAxP7rpe2viCdcfFpWEEWUiA/2rDnlrSudVq7NFP6H1QnPc\n5WlL2kHdBpLgDo631t3840E4zrfPeaRq4CMDu+nJexS+pOIzgtovQon89YC8a2B4\nX1Pp4gKN3QKBgQCrfXH/p3IOAwcl9qedyJG8W0mzMtbijnV9rxHK/4DZO+SRkvqk\nGdgpRH9CQfbgcfPOdyndNNyTcm3qGL3reJNlJBp69kPao+exQFF3q8GBbKev6ybP\n3v2EVcABVW/IjywRLRzkaItXm6PYg3N8s6IA7Wx5NQHnTzSqtI1k2NgjEwKBgENP\nxiFKKnoxRlvDDuV7kYwsrM/AH27fkji8HMb2r0e3MEiwWejIgp0RrxAScC5UQid0\nvnciboMeTfaWUq7Tsw3NQz43Ke1v0dH1K+mZvR46fYj3gB9AzZ6pTncWTuhAJDAH\nA46teSIEolaKSNR+3sTkg3KVUhqAMck3k5C7x1+NAoGAeJMUivx638QkWWOAAmW7\n2q3Pb6LAek1NR1/9k6a7rOU01GK28hnMd+wevAiI43XpJ7c9JW42gVdXdUvgpxsj\n4JGRRxkn4Ic36d+pH0OJSfRL/PmJB5rLli86O9hCEYBPlzz6WEAqa3X5Ek478S7H\nbH7sFs0UYIzCGx7dFoMcglQ=\n-----END PRIVATE KEY-----\n";

/// An ES256 CA for the two leaves below, whose common name is
/// only control characters ("\n", U+202E and "\t"), so it
/// has nothing left to show.
const BLANK_NAME_CA: &str = "-----BEGIN CERTIFICATE-----\nMIIBrTCCAVKgAwIBAgIUXrsW2a5fGEGoi56xJv/39nlO8O8wCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMDEwMDAwMDBaGA8yMTI2MTAxODAwMDAwMFowMzEZMBcGA1UE\nAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3AgVGVzdDBZMBMG\nByqGSM49AgEGCCqGSM49AwEHA0IABF90xEHbT8lU5VQL78nRk9A1hG0Pm3INKOiZ\nAG/ymzMrpdVOdKZ4058j3XVHAWUHWKK0Gr7W9nOV0/mQ5hWcQqqjQjBAMA8GA1Ud\nEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgEGMB0GA1UdDgQWBBRL7kDQdmlaI6mH\ntmY2MzZNRSsqaTAKBggqhkjOPQQDAgNJADBGAiEAlmMqmMs72ZOoYnrKeq/+g2mE\nwdw62dSif3aeUmVn/OECIQCcf3Jn/AD6fuQ0dvtqYIT3PcpBzJMSm+nsYeLFH2vR\nqQ==\n-----END CERTIFICATE-----\n";
/// A leaf with that common name and the organization
/// "Crabcrop Test".
const BLANK_NAME_LEAF: &str = "-----BEGIN CERTIFICATE-----\nMIIB0jCCAXqgAwIBAgIUSG6+pXgi+wk92XLpnOBx+tsteHowCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMDEwMDAwMDBaGA8yMTI2MTAxODAwMDAwMFowKDEOMAwGA1UE\nAwwFCuKArgkxFjAUBgNVBAoMDUNyYWJjcm9wIFRlc3QwWTATBgcqhkjOPQIBBggq\nhkjOPQMBBwNCAAQ+UPTwwYmV8NaJK08eHzxBoi129afV2B7vWnn/G8ddHUadyL5K\njm2XBHmHABFROzFfXYpvmA2reJj/AmF+LofJo3UwczAMBgNVHRMBAf8EAjAAMA4G\nA1UdDwEB/wQEAwIHgDATBgNVHSUEDDAKBggrBgEFBQcDBDAdBgNVHQ4EFgQUitMg\nEykF5Ze/dmCBvRfPVrX12RowHwYDVR0jBBgwFoAUS+5A0HZpWiOph7ZmNjM2TUUr\nKmkwCgYIKoZIzj0EAwIDRgAwQwIgNfA3IY4OeTybiY/OQNLDgxIGxvb8OEdkVz/h\nRcsJoIcCH2ZnMxKdEVbMVzqn/YzCglmXiZi+3lS45bFmKSnvqDU=\n-----END CERTIFICATE-----\n";
/// A leaf with that common name and no organization.
const BLANK_NAME_NO_ORG_LEAF: &str = "-----BEGIN CERTIFICATE-----\nMIIBvTCCAWKgAwIBAgIUe/ifwfIhNrfw04xIcxRvTIAaOzswCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMDEwMDAwMDBaGA8yMTI2MTAxODAwMDAwMFowEDEOMAwGA1UE\nAwwFCuKArgkwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQ+UPTwwYmV8NaJK08e\nHzxBoi129afV2B7vWnn/G8ddHUadyL5Kjm2XBHmHABFROzFfXYpvmA2reJj/AmF+\nLofJo3UwczAMBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQEAwIHgDATBgNVHSUEDDAK\nBggrBgEFBQcDBDAdBgNVHQ4EFgQUitMgEykF5Ze/dmCBvRfPVrX12RowHwYDVR0j\nBBgwFoAUS+5A0HZpWiOph7ZmNjM2TUUrKmkwCgYIKoZIzj0EAwIDSQAwRgIhAPvp\nkWTi2LZrwL+rloRCnJSieXXxNtqYIj1XBpAZ4QsaAiEA7/23i1edljq1p4aXBx52\nzCuDDM8TrQSxhxezFIsLnCw=\n-----END CERTIFICATE-----\n";
/// The private key for both leaves.
const BLANK_NAME_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgoxEyJErFzqICYbqv\n1xhSht+UsVarHZH0YNqheu/Zuu6hRANCAAQ+UPTwwYmV8NaJK08eHzxBoi129afV\n2B7vWnn/G8ddHUadyL5Kjm2XBHmHABFROzFfXYpvmA2reJj/AmF+LofJ\n-----END PRIVATE KEY-----\n";

/// The long-name identity, proven by the boot probe like any
/// other.
pub(super) fn long_name_signer() -> ProvenanceSigner {
    ProvenanceSigner::from_pem(
        LONG_NAME_CHAIN.as_bytes().to_vec(),
        LONG_NAME_KEY.as_bytes().to_vec(),
        None,
    )
    .expect("the long-name identity signs")
}

/// Whether `name` is one line of at most 64 characters, as
/// meta reports a signer.
fn is_one_bounded_line(name: &str) -> bool {
    name.chars().count() <= 64
        && !name
            .chars()
            .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
}

#[test]
fn a_signer_name_reads_back_as_one_bounded_line() {
    // Issue #595: the summary's signer is the name the
    // certificate gives, chosen by whoever made the file.
    // Meta reports it as one line of at most 64 characters,
    // ending in a marker when it was cut. The boot line's
    // name comes from the same readback.
    let signer = long_name_signer();
    let signed = signer
        .sign(
            SELF_TEST_PNG,
            "image/png",
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .expect("signing works");
    let summary = read_manifest(&signed, "image/png").expect("the output carries a manifest");
    assert_eq!(summary.state, "valid");
    let name = summary.signer.expect("the manifest names its signer");
    assert!(is_one_bounded_line(&name), "{name:?}");
    let first_63: String = LONG_NAME.chars().filter(|&c| c != '\n').take(63).collect();
    assert_eq!(name, format!("{first_63}…"));
    assert_eq!(signer.name, name);
}

#[test]
fn a_signer_name_loses_control_characters_and_stops_at_64_characters() {
    // Issue #595. A plain name passes unchanged.
    assert_eq!(
        display_signer("crabcrop test signer").as_deref(),
        Some("crabcrop test signer")
    );
    // Control characters (C0, DEL and C1), line and paragraph
    // separators, and bidirectional controls are dropped.
    assert_eq!(
        display_signer("a\nb\tc\r\0d\u{7f}e\u{85}f\u{2028}g\u{2029}h\u{202e}i\u{2066}j\u{200f}k")
            .as_deref(),
        Some("abcdefghijk")
    );
    // 64 characters is the bound; a dropped character does
    // not count toward it.
    let at_bound = "x".repeat(64);
    assert_eq!(
        display_signer(&at_bound).as_deref(),
        Some(at_bound.as_str())
    );
    assert_eq!(
        display_signer(&"x\n".repeat(64)).as_deref(),
        Some(at_bound.as_str())
    );
    // Past it, the first 63 characters and a marker.
    let cut = format!("{}…", "x".repeat(63));
    assert_eq!(
        display_signer(&"x".repeat(65)).as_deref(),
        Some(cut.as_str())
    );
    // The bound counts characters, not bytes, and a cut
    // never splits one.
    let wide = format!("{}…", "é".repeat(63));
    assert_eq!(
        display_signer(&"é".repeat(70)).as_deref(),
        Some(wide.as_str())
    );
    assert_eq!(
        display_signer(&"é".repeat(64)).as_deref(),
        Some("é".repeat(64).as_str())
    );
    // Tag characters (U+E0000 to U+E007F) display as nothing,
    // but a model still reads them, so they are dropped too.
    assert_eq!(
        display_signer("a\u{E0001}b\u{E0020}\u{E0041}\u{E007F}c").as_deref(),
        Some("abc")
    );
    // A name with nothing left is no name.
    assert_eq!(display_signer("\n\t\u{202e}\u{E0049}"), None);
    assert_eq!(display_signer(""), None);
}

#[test]
fn a_signer_falls_back_to_the_issuer_when_its_name_has_nothing_to_show() {
    // The common name comes first, and the issuer (c2pa-rs
    // fills it from the certificate's organization) stands in
    // when the name is absent or has nothing left once
    // cleaned. Both are cleaned the same way.
    assert_eq!(
        signer_name(Some("crabcrop test signer"), Some("Crabcrop Test")).as_deref(),
        Some("crabcrop test signer")
    );
    assert_eq!(
        signer_name(None, Some("Crabcrop Test")).as_deref(),
        Some("Crabcrop Test")
    );
    assert_eq!(
        signer_name(Some("\n\u{202e}\t"), Some("Crabcrop Test")).as_deref(),
        Some("Crabcrop Test")
    );
    assert_eq!(
        signer_name(Some(""), Some("Crabcrop\nTest")).as_deref(),
        Some("CrabcropTest")
    );
    assert_eq!(signer_name(Some("\n"), Some("\u{2028}")), None);
    assert_eq!(signer_name(Some("\n"), None), None);
    assert_eq!(signer_name(None, None), None);
}

#[test]
fn a_certificate_whose_name_has_nothing_to_show_boots_under_its_organization() {
    // A common name of control characters alone has nothing
    // left once cleaned. The boot probe then reads the
    // organization, as a summary does, and boot goes on.
    let signer = ProvenanceSigner::from_pem(
        format!("{BLANK_NAME_LEAF}{BLANK_NAME_CA}").into_bytes(),
        BLANK_NAME_KEY.as_bytes().to_vec(),
        None,
    )
    .expect("the identity signs");
    assert_eq!(signer.name, "Crabcrop Test");
    // With no organization either, the identity names no
    // signer, and boot says why.
    let error = ProvenanceSigner::from_pem(
        format!("{BLANK_NAME_NO_ORG_LEAF}{BLANK_NAME_CA}").into_bytes(),
        BLANK_NAME_KEY.as_bytes().to_vec(),
        None,
    )
    .expect_err("an identity with no name to show fails boot");
    assert!(
        error.contains("named no signer")
            && error.contains("common name and organization are missing"),
        "{error}"
    );
}

#[test]
fn an_rsa_identity_settles_on_ps256() {
    // Issue #600: an RSA PKCS#8 key loads under every PS
    // variant, and the search settles on the first, PS256.
    // Boot names the algorithm and warns for an RSA key.
    let signer = ProvenanceSigner::from_pem(
        RSA_CHAIN.as_bytes().to_vec(),
        RSA_KEY.as_bytes().to_vec(),
        None,
    )
    .expect("the RSA identity signs");
    assert_eq!(signer.alg(), SigningAlg::Ps256);
    assert_eq!(signer.name, "crabcrop test RSA signer");
    assert_eq!(tests::test_signer().alg(), SigningAlg::Es256);
}
