//! Fetching source images over HTTP(S).
//!
//! [`Loader`] fetches a URL with bounded time, bounded size, and a
//! bounded number of redirects. Anything that is not a successful
//! image response becomes a [`LoadError`]; the bytes are returned
//! untouched for the image engine to decode.
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
use std::time::Duration;

use reqwest::{Client, Url, header, redirect};

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

    /// Fetch `url` and return the response body.
    pub async fn load(&self, url: &str) -> Result<Vec<u8>, LoadError> {
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
            let mut resp = self
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
            return Ok(body);
        }
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
}
