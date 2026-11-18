//! AWS Signature Version 4 signing, hand-rolled on the tree's
//! existing `hmac` and `sha2` crates. Used to fetch from S3 and
//! from Google Cloud Storage's S3-compatible XML API, both of
//! which accept this scheme.
//!
//! The signer is deliberately small: one request at a time,
//! headers in, an `Authorization` header value out. It is
//! validated below against the worked examples from the AWS
//! documentation, signature and all: the IAM one, and the S3
//! ones for an object read, an object write that signs its
//! payload's hash, and bucket requests with query strings
//! (issue #709). The test fake S3 in `s3store` checks every
//! request it serves against its own reading of the scheme.

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

type HmacSha256 = Hmac<Sha256>;

/// The SHA-256 of the empty string: the payload hash of every
/// GET this service signs.
pub const EMPTY_PAYLOAD_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// A static credential pair (plus the optional session token).
/// The secret key and the session token are zeroed when it is
/// dropped.
#[derive(Clone, PartialEq, Eq)]
pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
}

impl Zeroize for Credentials {
    fn zeroize(&mut self) {
        self.secret_key.zeroize();
        self.session_token.zeroize();
    }
}

impl Drop for Credentials {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl ZeroizeOnDrop for Credentials {}

impl std::fmt::Debug for Credentials {
    /// The access key ID prints, since it names the identity
    /// and travels in every signed request's `Authorization`
    /// header anyway; the secret key and the session token
    /// print as `[redacted]` (issue #455).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("access_key", &self.access_key)
            .field("secret_key", &crate::redact::Redacted)
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| crate::redact::Redacted),
            )
            .finish()
    }
}

/// One request to sign. `path` is the percent-encoded path
/// exactly as it will be sent (SigV4 for S3 signs the
/// encoded-once form); `headers` are the headers to include in
/// the signature, host among them.
pub struct SignRequest<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub query: &'a [(&'a str, &'a str)],
    pub headers: &'a [(&'a str, &'a str)],
    pub region: &'a str,
    pub service: &'a str,
    pub payload_sha256: &'a str,
    /// Signing timestamp, `YYYYMMDDTHHMMSSZ` (see [`amz_date`]).
    pub amz_date: &'a str,
}

/// Percent-encode a string with the RFC 3986 unreserved set
/// kept (`A-Za-z0-9-._~`); everything else becomes `%XX` over
/// the UTF-8 bytes. Used for canonical query strings.
pub fn uri_encode(value: &str) -> String {
    encode_bytes(value.as_bytes(), false)
}

/// Percent-encode a URL path segment by segment, keeping the
/// `/` separators: how object keys are encoded for S3 URLs.
pub fn encode_path(path: &str) -> String {
    encode_bytes(path.as_bytes(), true)
}

fn encode_bytes(bytes: &[u8], keep_slash: bool) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        let keep = b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'.' | b'_' | b'~')
            || (keep_slash && b == b'/');
        if keep {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Format unix seconds as an SigV4 timestamp,
/// `YYYYMMDDTHHMMSSZ` in UTC.
pub fn amz_date(unix_secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(unix_secs);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

/// Break unix seconds into UTC civil parts. The days-to-date
/// conversion is the standard era arithmetic (Howard
/// Hinnant's `civil_from_days`).
fn civil(unix_secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = (unix_secs / 86_400) as i64;
    let secs = unix_secs % 86_400;
    let (h, mi, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m, d, h, mi, s)
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// A header value as the canonical request holds it: no
/// leading or trailing space, and every run of spaces inside
/// it one space. The store computes the same form from the
/// header it receives, so a value with a double space signed
/// as it stood would not verify.
fn trim_all(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Sign `req` and return the `Authorization` header value.
pub fn authorization(creds: &Credentials, req: &SignRequest<'_>) -> String {
    // Canonical query: encoded, sorted by key then value.
    let mut pairs: Vec<(String, String)> = req
        .query
        .iter()
        .map(|(k, v)| (uri_encode(k), uri_encode(v)))
        .collect();
    pairs.sort();
    let canonical_query = pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&");

    // Canonical headers: lowercase names, values trimmed with
    // each inner run of spaces made one (the scheme's
    // Trimall), sorted by name.
    let mut headers: Vec<(String, String)> = req
        .headers
        .iter()
        .map(|(k, v)| (k.to_lowercase(), trim_all(v)))
        .collect();
    headers.sort();
    let canonical_headers = headers
        .iter()
        .map(|(k, v)| format!("{k}:{v}\n"))
        .collect::<String>();
    let signed_headers = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        req.method,
        req.path,
        canonical_query,
        canonical_headers,
        signed_headers,
        req.payload_sha256
    );

    let date = &req.amz_date[..8];
    let scope = format!("{}/{}/{}/aws4_request", date, req.region, req.service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        req.amz_date,
        scope,
        sha256_hex(canonical_request.as_bytes())
    );

    // The signing key and each key it is derived through are
    // zeroed when they are dropped, like the secret key.
    let k_secret = Zeroizing::new(format!("AWS4{}", creds.secret_key));
    let k_date = Zeroizing::new(hmac(k_secret.as_bytes(), date.as_bytes()));
    let k_region = Zeroizing::new(hmac(&k_date, req.region.as_bytes()));
    let k_service = Zeroizing::new(hmac(&k_region, req.service.as_bytes()));
    let k_signing = Zeroizing::new(hmac(&k_service, b"aws4_request"));
    let signature = hex::encode(hmac(&k_signing, string_to_sign.as_bytes()));

    format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        creds.access_key, scope, signed_headers, signature
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_zero_their_secrets() {
        // What dropping a credential pair runs: the secret key
        // and the session token are wiped, and the access key
        // ID, which travels in every request, is kept.
        let mut creds = Credentials {
            access_key: "AKIDEXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: Some("session-token".to_string()),
        };
        creds.zeroize();
        assert_eq!(creds.secret_key, "");
        assert_eq!(creds.session_token, None);
        assert_eq!(creds.access_key, "AKIDEXAMPLE");
    }

    #[test]
    fn dates_format_in_utc() {
        assert_eq!(amz_date(0), "19700101T000000Z");
        // The timestamp of the AWS documentation example below.
        assert_eq!(amz_date(1_440_938_160), "20150830T123600Z");
        // A leap day and a year boundary.
        assert_eq!(amz_date(1_582_934_400), "20200229T000000Z");
        assert_eq!(amz_date(1_609_459_199), "20201231T235959Z");
        assert_eq!(amz_date(1_609_459_200), "20210101T000000Z");
    }

    #[test]
    fn encoding_keeps_unreserved_and_slashes_in_paths() {
        assert_eq!(uri_encode("a b+c/d"), "a%20b%2Bc%2Fd");
        assert_eq!(encode_path("a b/c+d/e~f"), "a%20b/c%2Bd/e~f");
        assert_eq!(encode_path("plain-key_1.2.jpg"), "plain-key_1.2.jpg");
    }

    /// The worked example from the AWS SigV4 documentation:
    /// signing a ListUsers request must reproduce the published
    /// signature exactly.
    #[test]
    fn matches_the_aws_documentation_vector() {
        let creds = Credentials {
            access_key: "AKIDEXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let auth = authorization(
            &creds,
            &SignRequest {
                method: "GET",
                path: "/",
                query: &[("Action", "ListUsers"), ("Version", "2010-05-08")],
                headers: &[
                    (
                        "Content-Type",
                        "application/x-www-form-urlencoded; charset=utf-8",
                    ),
                    ("Host", "iam.amazonaws.com"),
                    ("X-Amz-Date", "20150830T123600Z"),
                ],
                region: "us-east-1",
                service: "iam",
                payload_sha256: EMPTY_PAYLOAD_SHA256,
                amz_date: "20150830T123600Z",
            },
        );
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, \
             SignedHeaders=content-type;host;x-amz-date, \
             Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
        );
    }

    /// The credentials of the Amazon S3 examples in "Signature
    /// Calculations for the Authorization Header: Transferring
    /// Payload in a Single Chunk" (AWS Signature Version 4).
    fn s3_example_creds() -> Credentials {
        Credentials {
            access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        }
    }

    /// Sign one of the S3 examples: bucket `examplebucket`,
    /// `us-east-1`, 24 May 2013 at midnight.
    fn s3_example(
        method: &str,
        path: &str,
        query: &[(&str, &str)],
        extra_headers: &[(&str, &str)],
        payload_sha256: &str,
    ) -> String {
        let mut headers = vec![
            ("Host", "examplebucket.s3.amazonaws.com"),
            ("x-amz-date", "20130524T000000Z"),
            ("x-amz-content-sha256", payload_sha256),
        ];
        headers.extend_from_slice(extra_headers);
        authorization(
            &s3_example_creds(),
            &SignRequest {
                method,
                path,
                query,
                headers: &headers,
                region: "us-east-1",
                service: "s3",
                payload_sha256,
                amz_date: "20130524T000000Z",
            },
        )
    }

    /// The S3 examples' Authorization value for these signed
    /// headers and signature (the examples separate the fields
    /// with a bare comma; this signer, like the AWS SDKs, adds
    /// a space, which S3 accepts either way).
    fn s3_expected(signed_headers: &str, signature: &str) -> String {
        format!(
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders={signed_headers}, Signature={signature}"
        )
    }

    #[test]
    fn matches_the_s3_get_object_example() {
        // Issue #709: the signer was proven only against the
        // IAM example, which signs no S3 request. A ranged GET
        // of an object, the shape of every shared-tier read.
        let auth = s3_example(
            "GET",
            "/test.txt",
            &[],
            &[("Range", "bytes=0-9")],
            EMPTY_PAYLOAD_SHA256,
        );
        assert_eq!(
            auth,
            s3_expected(
                "host;range;x-amz-content-sha256;x-amz-date",
                "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
            )
        );
    }

    #[test]
    fn matches_the_s3_put_object_example() {
        // A write that signs the hash of its payload, under a
        // key the path encodes (`$` is `%24`), with a header
        // whose value has inner spaces.
        let body = b"Welcome to Amazon S3.";
        let payload = sha256_hex(body);
        assert_eq!(
            payload,
            "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
        );
        let path = format!("/{}", encode_path("test$file.text"));
        assert_eq!(path, "/test%24file.text");
        let auth = s3_example(
            "PUT",
            &path,
            &[],
            &[
                ("Date", "Fri, 24 May 2013 00:00:00 GMT"),
                ("x-amz-storage-class", "REDUCED_REDUNDANCY"),
            ],
            &payload,
        );
        assert_eq!(
            auth,
            s3_expected(
                "date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class",
                "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"
            )
        );
    }

    #[test]
    fn matches_the_s3_bucket_examples_with_query_strings() {
        // A sub-resource with no value signs as `lifecycle=`,
        // and a listing's parameters sort by name: the shape
        // of the purge walk's and the tombstone refresh's
        // ListObjectsV2 pages.
        let lifecycle = s3_example("GET", "/", &[("lifecycle", "")], &[], EMPTY_PAYLOAD_SHA256);
        assert_eq!(
            lifecycle,
            s3_expected(
                "host;x-amz-content-sha256;x-amz-date",
                "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"
            )
        );
        let listing = s3_example(
            "GET",
            "/",
            &[("prefix", "J"), ("max-keys", "2")],
            &[],
            EMPTY_PAYLOAD_SHA256,
        );
        assert_eq!(
            listing,
            s3_expected(
                "host;x-amz-content-sha256;x-amz-date",
                "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
            )
        );
    }

    #[test]
    fn header_values_sign_with_inner_spaces_collapsed() {
        // The scheme's Trimall: the store signs a header value
        // trimmed, each inner run of spaces made one, so the
        // signer must too. Signing the PUT example's Date with
        // padding gives the published signature.
        let payload = sha256_hex(b"Welcome to Amazon S3.");
        let auth = s3_example(
            "PUT",
            "/test%24file.text",
            &[],
            &[
                ("Date", "  Fri,  24 May   2013 00:00:00 GMT "),
                ("x-amz-storage-class", "REDUCED_REDUNDANCY"),
            ],
            &payload,
        );
        assert!(
            auth.ends_with(
                "Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"
            ),
            "{auth}"
        );
    }
}
