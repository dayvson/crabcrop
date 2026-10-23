//! Fetching source images.
//!
//! [`Loader`] fetches a URL with bounded time, bounded size, and a
//! bounded number of redirects. Anything that is not a successful
//! image response becomes a [`LoadError`]; the bytes are returned
//! untouched for the image engine to decode.
//!
//! Four source kinds, chosen by the URL scheme:
//!
//! - `http://` and `https://`, under the [`SourcePolicy`] below.
//! - `s3://bucket/key`, fetched from AWS S3 (virtual-hosted) or
//!   from a configured S3-compatible endpoint (path-style), signed
//!   with SigV4 when credentials are configured and unsigned
//!   otherwise.
//! - `gs://bucket/key`, the same treatment against Google Cloud
//!   Storage's S3-compatible XML API.
//! - `file://...`, local files under a configured root,
//!   canonicalized per request so paths cannot escape it.
//!
//! Bucket sources are gated by a bucket allowlist rather than the
//! host policy: the endpoint is operator configuration, not
//! request input. Store responses are not redirected; a 3xx from
//! a store surfaces as its status.
//!
//! A [`SourcePolicy`] can restrict which sources may be fetched: an
//! allowlist of host suffixes, and blocking of hosts that resolve to
//! loopback, private, link-local, or unspecified addresses. Both
//! checks run before any connection is made, and again on every
//! redirect hop (redirects are followed manually for exactly that
//! reason).
//!
//! Known limit: the private-address check resolves the host and
//! inspects the answers, but the HTTP client resolves again when it
//! connects. A hostile authoritative DNS server could answer
//! differently the second time (DNS rebinding). Closing that gap
//! needs a resolver that pins the checked addresses, which the
//! client does not expose; the check still stops direct internal
//! URLs, which is the common exposure.

use std::net::IpAddr;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::{Client, Url, header, redirect};

use crate::sigv4::{self, Credentials};

/// Default cap on response bodies: 10 MiB.
const DEFAULT_MAX_BYTES: u64 = 10 * 1024 * 1024;
/// Default total request timeout: 10 seconds.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
/// Maximum redirects followed per request.
const MAX_REDIRECTS: usize = 5;

/// What went wrong while fetching a source image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// DNS, connect, TLS, or body-stream failure. Carries the cause.
    Network(String),
    /// The server answered with a non-2xx status. Carries the code.
    BadStatus(u16),
    /// The response exceeded the byte limit. Carries the limit.
    TooLarge { limit: u64 },
    /// The `Content-Type` header was present but not an image type.
    /// Carries the header value when it was readable. A missing header
    /// is allowed; the image engine rejects undecodable bytes later.
    WrongContentType(Option<String>),
    /// The request took longer than the timeout.
    Timeout,
    /// The source policy refused this URL. Carries the reason.
    Forbidden(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Network(cause) => write!(f, "network error: {cause}"),
            LoadError::BadStatus(code) => write!(f, "source answered HTTP {code}"),
            LoadError::TooLarge { limit } => {
                write!(f, "source exceeded the {limit}-byte limit")
            }
            LoadError::WrongContentType(Some(ct)) => {
                write!(f, "source content type is not an image: {ct}")
            }
            LoadError::WrongContentType(None) => {
                write!(f, "source content type is not readable")
            }
            LoadError::Timeout => write!(f, "source fetch timed out"),
            LoadError::Forbidden(reason) => {
                write!(f, "source refused by policy: {reason}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Which sources the loader may fetch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourcePolicy {
    /// Host suffixes an URL's host must match (exactly, or as a
    /// subdomain). Empty means no allowlist.
    pub allowed_suffixes: Vec<String>,
    /// Refuse hosts resolving to non-public addresses.
    pub block_private: bool,
}

impl SourcePolicy {
    /// Build a policy, normalizing suffixes (lowercase, no leading
    /// dot, no blanks).
    pub fn new(allowed_suffixes: Vec<String>, block_private: bool) -> Self {
        let allowed_suffixes = allowed_suffixes
            .iter()
            .map(|s| s.trim().trim_start_matches('.').to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            allowed_suffixes,
            block_private,
        }
    }
}

/// Fetches source image bytes over HTTP(S).
#[derive(Debug, Clone)]
pub struct Loader {
    client: Client,
    max_bytes: u64,
    policy: SourcePolicy,
    sources: SourceConfig,
}

/// Configuration for the non-HTTP source kinds: object stores
/// and local files. Defaults mean: AWS's real endpoints, no
/// credentials (unsigned fetches), no bucket allowlist, and
/// file sources off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceConfig {
    /// Base URL of an S3-compatible store, e.g.
    /// `http://localhost:9000` for MinIO. `None` means AWS.
    pub s3_endpoint: Option<String>,
    /// AWS region for endpoint derivation and signing.
    pub s3_region: String,
    /// Ambient AWS credentials, when configured.
    pub aws_creds: Option<Credentials>,
    /// Base URL of the GCS XML API (or a fake, in tests).
    pub gcs_endpoint: String,
    /// Credential scope region for GCS signing.
    pub gcs_region: String,
    /// GCS HMAC credentials, when configured.
    pub gcs_creds: Option<Credentials>,
    /// Bucket names the bucket schemes may read. An entry of
    /// `*` allows every bucket. Boot refuses the combination of
    /// configured credentials and an empty list (audit L-1).
    pub allowed_buckets: Vec<String>,
    /// Canonicalized root that `file://` sources must stay
    /// under. `None` disables file sources.
    pub file_root: Option<PathBuf>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            s3_endpoint: None,
            s3_region: "us-east-1".to_string(),
            aws_creds: None,
            gcs_endpoint: "https://storage.googleapis.com".to_string(),
            gcs_region: "auto".to_string(),
            gcs_creds: None,
            allowed_buckets: Vec::new(),
            file_root: None,
        }
    }
}

/// Which object store a bucket URL names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreKind {
    S3,
    Gcs,
}

impl Loader {
    /// Build a loader with an explicit byte limit and total timeout.
    pub fn new(max_bytes: u64, timeout: Duration) -> Result<Self, LoadError> {
        let client = Client::builder()
            .timeout(timeout)
            // Redirects are followed manually in `load` so the
            // source policy is re-checked on every hop.
            .redirect(redirect::Policy::none())
            .build()
            .map_err(|e| LoadError::Network(e.to_string()))?;
        Ok(Self {
            client,
            max_bytes,
            policy: SourcePolicy::default(),
            sources: SourceConfig::default(),
        })
    }

    /// Build a loader with the default byte limit and timeout.
    pub fn with_defaults() -> Result<Self, LoadError> {
        Self::new(DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT)
    }

    /// Set the source policy (allowlist and private-address
    /// blocking). The default policy allows every source.
    pub fn with_policy(mut self, policy: SourcePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the bucket and file source configuration.
    pub fn with_sources(mut self, sources: SourceConfig) -> Self {
        self.sources = sources;
        self
    }

    /// Fetch `url` and return the body bytes, dispatching on
    /// the source scheme.
    pub async fn load(&self, url: &str) -> Result<Vec<u8>, LoadError> {
        if let Some(rest) = url.strip_prefix("s3://") {
            let (bucket, key) = split_bucket_key(rest, "s3")?;
            return self.load_object(StoreKind::S3, bucket, key).await;
        }
        if let Some(rest) = url.strip_prefix("gs://") {
            let (bucket, key) = split_bucket_key(rest, "gs")?;
            return self.load_object(StoreKind::Gcs, bucket, key).await;
        }
        if url.starts_with("file://") {
            return self.load_file(url).await;
        }
        self.load_http(url).await
    }

    async fn load_http(&self, url: &str) -> Result<Vec<u8>, LoadError> {
        let mut parsed =
            Url::parse(url).map_err(|e| LoadError::Network(format!("invalid url: {e}")))?;
        let mut hops = 0;
        loop {
            if parsed.scheme() != "http" && parsed.scheme() != "https" {
                return Err(LoadError::Network(format!(
                    "unsupported scheme: {}",
                    parsed.scheme()
                )));
            }
            self.check_policy(&parsed).await?;
            let resp = self
                .client
                .get(parsed.clone())
                .send()
                .await
                .map_err(map_request_error)?;
            let status = resp.status();
            if status.is_redirection() {
                if hops >= MAX_REDIRECTS {
                    return Err(LoadError::Network("too many redirects".to_string()));
                }
                hops += 1;
                let location = resp
                    .headers()
                    .get(header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or(LoadError::BadStatus(status.as_u16()))?;
                parsed = parsed
                    .join(location)
                    .map_err(|e| LoadError::Network(format!("invalid redirect: {e}")))?;
                continue;
            }
            return self.read_checked_body(resp).await;
        }
    }

    /// Status, content-type, and bounded-body handling shared by
    /// the HTTP path and the object stores. A non-2xx status
    /// (including a store redirect, which is not followed) is an
    /// error.
    async fn read_checked_body(&self, mut resp: reqwest::Response) -> Result<Vec<u8>, LoadError> {
        let status = resp.status();
        if !status.is_success() {
            return Err(LoadError::BadStatus(status.as_u16()));
        }
        if let Some(value) = resp.headers().get(header::CONTENT_TYPE) {
            let content_type = value.to_str().ok();
            let is_image = content_type.is_some_and(|ct| ct.starts_with("image/"));
            if !is_image {
                return Err(LoadError::WrongContentType(
                    content_type.map(str::to_string),
                ));
            }
        }
        let mut body = Vec::new();
        loop {
            let chunk = resp.chunk().await.map_err(map_request_error)?;
            let Some(bytes) = chunk else { break };
            body.extend_from_slice(&bytes);
            if body.len() as u64 > self.max_bytes {
                return Err(LoadError::TooLarge {
                    limit: self.max_bytes,
                });
            }
        }
        Ok(body)
    }

    /// Fetch one object from S3 or GCS, signing with SigV4 when
    /// credentials are configured for the store.
    async fn load_object(
        &self,
        kind: StoreKind,
        bucket: &str,
        key: &str,
    ) -> Result<Vec<u8>, LoadError> {
        if !self.sources.allowed_buckets.is_empty()
            && !self
                .sources
                .allowed_buckets
                .iter()
                .any(|b| b == bucket || b == "*")
        {
            return Err(LoadError::Forbidden(format!(
                "bucket {bucket} is not an allowed source"
            )));
        }
        let encoded_key = sigv4::encode_path(key);
        let (url_string, creds, region) = match kind {
            StoreKind::S3 => {
                let url = match &self.sources.s3_endpoint {
                    Some(endpoint) => format!("{endpoint}/{bucket}/{encoded_key}"),
                    None => format!(
                        "https://{bucket}.s3.{}.amazonaws.com/{encoded_key}",
                        self.sources.s3_region
                    ),
                };
                (
                    url,
                    self.sources.aws_creds.as_ref(),
                    self.sources.s3_region.as_str(),
                )
            }
            StoreKind::Gcs => (
                format!("{}/{bucket}/{encoded_key}", self.sources.gcs_endpoint),
                self.sources.gcs_creds.as_ref(),
                self.sources.gcs_region.as_str(),
            ),
        };
        let url = Url::parse(&url_string)
            .map_err(|e| LoadError::Network(format!("invalid store url: {e}")))?;
        let mut request = self.client.get(url.clone());
        if let Some(creds) = creds {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let amz_date = sigv4::amz_date(now);
            let host = match url.port() {
                Some(port) => format!("{}:{port}", url.host_str().unwrap_or_default()),
                None => url.host_str().unwrap_or_default().to_string(),
            };
            let mut sign_headers: Vec<(&str, &str)> = vec![
                ("host", host.as_str()),
                ("x-amz-content-sha256", sigv4::EMPTY_PAYLOAD_SHA256),
                ("x-amz-date", amz_date.as_str()),
            ];
            if let Some(token) = &creds.session_token {
                sign_headers.push(("x-amz-security-token", token.as_str()));
            }
            let authorization = sigv4::authorization(
                creds,
                &sigv4::SignRequest {
                    method: "GET",
                    path: url.path(),
                    query: &[],
                    headers: &sign_headers,
                    region,
                    service: "s3",
                    payload_sha256: sigv4::EMPTY_PAYLOAD_SHA256,
                    amz_date: &amz_date,
                },
            );
            request = request
                .header("Authorization", authorization)
                .header("x-amz-date", amz_date.as_str())
                .header("x-amz-content-sha256", sigv4::EMPTY_PAYLOAD_SHA256);
            if let Some(token) = &creds.session_token {
                request = request.header("x-amz-security-token", token.as_str());
            }
        }
        let resp = request.send().await.map_err(map_request_error)?;
        self.read_checked_body(resp).await
    }

    /// Fetch a local file under the configured root. The
    /// candidate path is canonicalized before the containment
    /// check, so `..` segments and symlink escapes are refused.
    async fn load_file(&self, url: &str) -> Result<Vec<u8>, LoadError> {
        use tokio::io::AsyncReadExt;
        let Some(root) = &self.sources.file_root else {
            return Err(LoadError::Forbidden(
                "file sources are not configured".to_string(),
            ));
        };
        let parsed =
            Url::parse(url).map_err(|e| LoadError::Network(format!("invalid url: {e}")))?;
        match parsed.host_str() {
            None | Some("") | Some("localhost") => {}
            Some(host) => {
                return Err(LoadError::Forbidden(format!(
                    "file source host {host} is not local"
                )));
            }
        }
        let decoded = percent_decode(parsed.path());
        if decoded.contains('\0') {
            return Err(LoadError::Network("invalid file source path".to_string()));
        }
        let candidate = root.join(decoded.trim_start_matches('/'));
        let canonical = tokio::fs::canonicalize(&candidate)
            .await
            .map_err(|e| LoadError::Network(format!("cannot read file source: {e}")))?;
        if !canonical.starts_with(root) {
            return Err(LoadError::Forbidden(
                "file source escapes the configured root".to_string(),
            ));
        }
        let mut file = tokio::fs::File::open(&canonical)
            .await
            .map_err(|e| LoadError::Network(format!("cannot read file source: {e}")))?;
        let mut body = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file
                .read(&mut buf)
                .await
                .map_err(|e| LoadError::Network(format!("cannot read file source: {e}")))?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
            if body.len() as u64 > self.max_bytes {
                return Err(LoadError::TooLarge {
                    limit: self.max_bytes,
                });
            }
        }
        Ok(body)
    }

    /// Run the source policy against `url`. No connection has been
    /// made yet when this runs.
    async fn check_policy(&self, url: &Url) -> Result<(), LoadError> {
        let host = url
            .host_str()
            .ok_or_else(|| LoadError::Forbidden("source URL has no host".to_string()))?;
        if !self.policy.allowed_suffixes.is_empty()
            && !self
                .policy
                .allowed_suffixes
                .iter()
                .any(|s| suffix_matches(host, s))
        {
            return Err(LoadError::Forbidden(format!(
                "host {host} is not an allowed source"
            )));
        }
        if self.policy.block_private {
            // IPv6 literals arrive bracketed from the URL parser.
            let bare = host.trim_start_matches('[').trim_end_matches(']');
            let ips: Vec<IpAddr> = match bare.parse::<IpAddr>() {
                Ok(ip) => vec![ip],
                Err(_) => tokio::net::lookup_host((host, 0))
                    .await
                    .map_err(|e| LoadError::Network(format!("dns lookup failed: {e}")))?
                    .map(|addr| addr.ip())
                    .collect(),
            };
            if let Some(ip) = ips.iter().find(|ip| is_non_public(ip)) {
                return Err(LoadError::Forbidden(format!(
                    "host {host} resolves to a non-public address ({ip})"
                )));
            }
        }
        Ok(())
    }
}

/// Whether `host` equals `suffix` or is a subdomain of it. The dot
/// boundary is what keeps `evil-example.com` from matching
/// `example.com`.
fn suffix_matches(host: &str, suffix: &str) -> bool {
    host == suffix || host.ends_with(&format!(".{suffix}"))
}

/// Loopback, private, link-local, and unspecified addresses are
/// never legitimate public image sources.
fn is_non_public(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_loopback() || ip.is_private() || ip.is_link_local() || ip.is_unspecified()
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || ip.is_unspecified()
        }
    }
}

/// Split the `bucket/key` remainder of a bucket URL. Both parts
/// are required; the key may itself contain slashes.
fn split_bucket_key<'a>(rest: &'a str, scheme: &str) -> Result<(&'a str, &'a str), LoadError> {
    let invalid = || {
        LoadError::Network(format!(
            "invalid {scheme} url: expected {scheme}://bucket/key"
        ))
    };
    let (bucket, key) = rest.split_once('/').ok_or_else(invalid)?;
    if bucket.is_empty() || key.is_empty() {
        return Err(invalid());
    }
    Ok((bucket, key))
}

/// Decode `%XX` escapes in a URL path. A literal `+` stays a
/// plus (paths are not form data); malformed escapes pass
/// through untouched.
fn percent_decode(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let digit = |c: u8| (c as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (digit(bytes[i + 1]), digit(bytes[i + 2])) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn map_request_error(e: reqwest::Error) -> LoadError {
    if e.is_timeout() {
        LoadError::Timeout
    } else {
        LoadError::Network(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio::time::sleep;

    /// What the test server does with its one connection.
    enum Respond {
        /// Write these raw HTTP response bytes, then close.
        Bytes(Vec<u8>),
        /// Hold the connection open (for timeout tests).
        Hang,
    }

    /// Spawn a one-shot HTTP server on 127.0.0.1. Returns the base URL.
    async fn spawn_server(respond: Respond) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            match respond {
                Respond::Bytes(bytes) => {
                    let _ = socket.write_all(&bytes).await;
                }
                Respond::Hang => {
                    sleep(Duration::from_secs(3600)).await;
                }
            }
        });
        format!("http://{addr}")
    }

    fn response(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
        let mut out =
            format!("HTTP/1.1 {status}\r\n{headers}Connection: close\r\n\r\n").into_bytes();
        out.extend_from_slice(body);
        out
    }

    fn ok_image_response() -> Vec<u8> {
        response(
            "200 OK",
            "Content-Type: image/jpeg\r\nContent-Length: 4\r\n",
            b"\xff\xd8\xff\xd9",
        )
    }

    fn fetch<'a>(
        loader: &'a Loader,
        base: &str,
    ) -> impl Future<Output = Result<Vec<u8>, LoadError>> + 'a {
        let url = format!("{base}/img.jpg");
        async move { loader.load(&url).await }
    }

    #[tokio::test]
    async fn loads_image_bytes() {
        let base = spawn_server(Respond::Bytes(ok_image_response())).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(fetch(&loader, &base).await.unwrap(), b"\xff\xd8\xff\xd9");
    }

    #[test]
    fn errors_render() {
        assert_eq!(
            LoadError::BadStatus(404).to_string(),
            "source answered HTTP 404"
        );
        assert_eq!(
            LoadError::TooLarge { limit: 10 }.to_string(),
            "source exceeded the 10-byte limit"
        );
    }

    #[tokio::test]
    async fn rejects_non_image_content_type() {
        let body = b"<html></html>";
        let resp = response(
            "200 OK",
            "Content-Type: text/html\r\nContent-Length: 13\r\n",
            body,
        );
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(
            fetch(&loader, &base).await,
            Err(LoadError::WrongContentType(Some("text/html".to_string())))
        );
    }

    #[tokio::test]
    async fn allows_missing_content_type() {
        // No Content-Type header: the loader proceeds and the image
        // engine is the backstop for undecodable bytes.
        let resp = response("200 OK", "Content-Length: 4\r\n", b"\xff\xd8\xff\xd9");
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(fetch(&loader, &base).await.unwrap(), b"\xff\xd8\xff\xd9");
    }

    #[tokio::test]
    async fn rejects_bad_status() {
        let resp = response("404 Not Found", "Content-Length: 0\r\n", b"");
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(fetch(&loader, &base).await, Err(LoadError::BadStatus(404)));
    }

    #[tokio::test]
    async fn enforces_size_limit() {
        let big = vec![0u8; 100];
        let resp = response(
            "200 OK",
            "Content-Type: image/jpeg\r\nContent-Length: 100\r\n",
            &big,
        );
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::new(10, Duration::from_secs(5)).unwrap();
        assert_eq!(
            fetch(&loader, &base).await,
            Err(LoadError::TooLarge { limit: 10 })
        );
    }

    #[tokio::test]
    async fn times_out_on_hanging_server() {
        let base = spawn_server(Respond::Hang).await;
        let loader = Loader::new(1024, Duration::from_millis(100)).unwrap();
        assert_eq!(fetch(&loader, &base).await, Err(LoadError::Timeout));
    }

    #[tokio::test]
    async fn follows_redirects() {
        let target = spawn_server(Respond::Bytes(ok_image_response())).await;
        let redirect_resp = response(
            "302 Found",
            &format!("Location: {target}/img.jpg\r\nContent-Length: 0\r\n"),
            b"",
        );
        let base = spawn_server(Respond::Bytes(redirect_resp)).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(fetch(&loader, &base).await.unwrap(), b"\xff\xd8\xff\xd9");
    }

    #[test]
    fn suffix_matching_respects_dot_boundaries() {
        assert!(suffix_matches("example.com", "example.com"));
        assert!(suffix_matches("cdn.example.com", "example.com"));
        assert!(!suffix_matches("evil-example.com", "example.com"));
        assert!(!suffix_matches("example.com.evil.com", "example.com"));
        assert!(!suffix_matches("ample.com", "example.com"));
    }

    #[test]
    fn non_public_addresses_are_detected() {
        for s in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.5",
            "192.168.1.1",
            "169.254.1.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fd00::1",
        ] {
            assert!(is_non_public(&s.parse().unwrap()), "{s}");
        }
        for s in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(!is_non_public(&s.parse().unwrap()), "{s}");
        }
    }

    #[tokio::test]
    async fn allowlist_accepts_listed_host_and_rejects_others() {
        let base = spawn_server(Respond::Bytes(ok_image_response())).await;
        let listed = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec!["127.0.0.1".to_string()], false));
        assert!(fetch(&listed, &base).await.is_ok());

        let base = spawn_server(Respond::Bytes(ok_image_response())).await;
        let unlisted = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec!["example.com".to_string()], false));
        let err = fetch(&unlisted, &base).await.unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
    }

    #[tokio::test]
    async fn private_blocking_rejects_loopback_when_enabled() {
        let base = spawn_server(Respond::Bytes(ok_image_response())).await;
        let blocking = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec![], true));
        let err = fetch(&blocking, &base).await.unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
        // The blocked attempt never connected, so the one-shot
        // server is still there for an unblocked loader.
        let open = Loader::with_defaults().unwrap();
        assert!(fetch(&open, &base).await.is_ok());
    }

    #[tokio::test]
    async fn private_blocking_rejects_literal_private_ips() {
        let blocking = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec![], true));
        for url in [
            "http://10.0.0.8/img.jpg",
            "http://192.168.0.20/img.jpg",
            "http://[::1]/img.jpg",
        ] {
            let err = blocking.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
        }
    }

    #[tokio::test]
    async fn redirect_hops_are_rechecked() {
        // The first host is allowlisted; its redirect target is not.
        let redirect_resp = response(
            "302 Found",
            "Location: http://10.0.0.8/img.jpg\r\nContent-Length: 0\r\n",
            b"",
        );
        let base = spawn_server(Respond::Bytes(redirect_resp)).await;
        let loader = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec!["127.0.0.1".to_string()], false));
        let err = fetch(&loader, &base).await.unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
    }

    /// Spawn a one-shot server that records the request head it
    /// receives, then replies with the ok image response.
    async fn spawn_capturing_server() -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        use tokio::io::AsyncReadExt;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink = captured.clone();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                let n = socket.read(&mut chunk).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            *sink.lock().unwrap() = String::from_utf8_lossy(&buf).into_owned();
            let _ = socket.write_all(&ok_image_response()).await;
        });
        (format!("http://{addr}"), captured)
    }

    fn test_creds() -> Credentials {
        Credentials {
            access_key: "AKIDTEST".to_string(),
            secret_key: "test-secret".to_string(),
            session_token: Some("tok123".to_string()),
        }
    }

    #[tokio::test]
    async fn s3_loads_unsigned_from_a_custom_endpoint() {
        let (base, captured) = spawn_capturing_server().await;
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            s3_endpoint: Some(base),
            ..Default::default()
        });
        let body = loader.load("s3://photos/dir/pic.jpg").await.unwrap();
        assert_eq!(body, b"\xff\xd8\xff\xd9");
        let head = captured.lock().unwrap().clone();
        let first = head.lines().next().unwrap().to_string();
        assert!(first.starts_with("GET /photos/dir/pic.jpg "), "{first}");
        assert!(!head.to_lowercase().contains("authorization:"), "{head}");
    }

    #[tokio::test]
    async fn s3_signs_when_credentials_are_configured() {
        let (base, captured) = spawn_capturing_server().await;
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            s3_endpoint: Some(base),
            aws_creds: Some(test_creds()),
            ..Default::default()
        });
        // The key's space is encoded for both the URL and the
        // signature.
        loader.load("s3://photos/a b.jpg").await.unwrap();
        let head = captured.lock().unwrap().clone();
        let lower = head.to_lowercase();
        let first = head.lines().next().unwrap().to_string();
        assert!(first.starts_with("GET /photos/a%20b.jpg "), "{first}");
        assert!(
            lower.contains("authorization: aws4-hmac-sha256 credential=akidtest/"),
            "{head}"
        );
        assert!(
            lower.contains(
                "signedheaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
            ),
            "{head}"
        );
        assert!(lower.contains("x-amz-date: "), "{head}");
        assert!(lower.contains("x-amz-security-token: tok123"), "{head}");
        assert!(lower.contains("signature="), "{head}");
    }

    #[tokio::test]
    async fn gs_loads_from_its_endpoint() {
        let (base, captured) = spawn_capturing_server().await;
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            gcs_endpoint: base,
            ..Default::default()
        });
        let body = loader.load("gs://media/pic.jpg").await.unwrap();
        assert_eq!(body, b"\xff\xd8\xff\xd9");
        let head = captured.lock().unwrap().clone();
        let first = head.lines().next().unwrap().to_string();
        assert!(first.starts_with("GET /media/pic.jpg "), "{first}");
    }

    #[tokio::test]
    async fn bucket_allowlist_gates_both_schemes() {
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            allowed_buckets: vec!["photos".to_string()],
            ..Default::default()
        });
        for url in ["s3://other/pic.jpg", "gs://other/pic.jpg"] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
        }
    }

    #[tokio::test]
    async fn malformed_bucket_urls_error() {
        let loader = Loader::with_defaults().unwrap();
        for url in ["s3://", "s3://bucketonly", "s3://bucket/", "gs:///key"] {
            assert!(loader.load(url).await.is_err(), "{url}");
        }
    }

    /// A scratch tree: `root/` holds the servable files, a
    /// secret sits next to it outside the root.
    fn make_file_tree() -> (PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "crabcrop-loader-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = base.join("root");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("pic.jpg"), b"rootpic").unwrap();
        std::fs::write(root.join("sub/inner.jpg"), b"inner").unwrap();
        std::fs::write(root.join("big.bin"), vec![7u8; 64]).unwrap();
        std::fs::write(base.join("secret.jpg"), b"secret").unwrap();
        let root = std::fs::canonicalize(&root).unwrap();
        (base, root)
    }

    fn file_loader(root: &std::path::Path, max_bytes: u64) -> Loader {
        Loader::new(max_bytes, Duration::from_secs(5))
            .unwrap()
            .with_sources(SourceConfig {
                file_root: Some(root.to_path_buf()),
                ..Default::default()
            })
    }

    #[tokio::test]
    async fn file_sources_load_under_the_root() {
        let (base, root) = make_file_tree();
        let loader = file_loader(&root, 1024);
        assert_eq!(loader.load("file:///pic.jpg").await.unwrap(), b"rootpic");
        assert_eq!(
            loader.load("file://localhost/sub/inner.jpg").await.unwrap(),
            b"inner"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn file_sources_cannot_escape_the_root() {
        let (base, root) = make_file_tree();
        // Symlinks inside the root pointing outside: the
        // canonicalization check is what refuses these.
        std::os::unix::fs::symlink(base.join("secret.jpg"), root.join("link.jpg")).unwrap();
        std::os::unix::fs::symlink(&base, root.join("linkdir")).unwrap();
        let loader = file_loader(&root, 1024);
        for url in ["file:///link.jpg", "file:///linkdir/secret.jpg"] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
        }
        // Dot-segment escapes are collapsed by URL parsing into
        // the root and fail there as missing files; either way
        // the secret is never served.
        for url in ["file:///../secret.jpg", "file:///sub/../../secret.jpg"] {
            assert!(loader.load(url).await.is_err(), "{url}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn file_sources_missing_unconfigured_and_oversized() {
        let (base, root) = make_file_tree();
        let loader = file_loader(&root, 1024);
        assert!(loader.load("file:///nope.jpg").await.is_err());
        let small = file_loader(&root, 8);
        assert_eq!(
            small.load("file:///big.bin").await,
            Err(LoadError::TooLarge { limit: 8 })
        );
        let unconfigured = Loader::with_defaults().unwrap();
        let err = unconfigured.load("file:///pic.jpg").await.unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn bucket_keys_and_percent_decoding_parse() {
        assert_eq!(
            split_bucket_key("b/dir/k.jpg", "s3").unwrap(),
            ("b", "dir/k.jpg")
        );
        assert!(split_bucket_key("b/", "s3").is_err());
        assert_eq!(percent_decode("/a%20b/%2e%2e/c+d"), "/a b/../c+d");
        assert_eq!(percent_decode("/plain%zz%"), "/plain%zz%");
    }
}
