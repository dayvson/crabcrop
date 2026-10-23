//! AWS Signature Version 4 signing, hand-rolled on the tree's
//! existing `hmac` and `sha2` crates. Used to fetch from S3 and
//! from Google Cloud Storage's S3-compatible XML API, both of
//! which accept this scheme.
//!
//! The signer is deliberately small: one GET-shaped request at a
//! time, headers in, an `Authorization` header value out. It is
//! validated below against the worked example from the AWS
//! documentation, signature and all.

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// The SHA-256 of the empty string: the payload hash of every
/// GET this service signs.
pub const EMPTY_PAYLOAD_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// A static credential pair (plus the optional session token).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
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

    // Canonical headers: lowercase names, trimmed values,
    // sorted by name.
    let mut headers: Vec<(String, String)> = req
        .headers
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.trim().to_string()))
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

    let k_date = hmac(
        format!("AWS4{}", creds.secret_key).as_bytes(),
        date.as_bytes(),
    );
    let k_region = hmac(&k_date, req.region.as_bytes());
    let k_service = hmac(&k_region, req.service.as_bytes());
    let k_signing = hmac(&k_service, b"aws4_request");
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
}
