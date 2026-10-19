//! Fetching source images over HTTP(S).
//!
//! [`Loader`] fetches a URL with bounded time, bounded size, and a
//! bounded number of redirects. Anything that is not a successful
//! image response becomes a [`LoadError`]; the bytes are returned
//! untouched for the image engine to decode.

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
}

/// Fetches source image bytes over HTTP(S).
#[derive(Debug, Clone)]
pub struct Loader {
    client: Client,
    max_bytes: u64,
}

impl Loader {
    /// Build a loader with an explicit byte limit and total timeout.
    pub fn new(max_bytes: u64, timeout: Duration) -> Result<Self, LoadError> {
        let client = Client::builder()
            .timeout(timeout)
            .redirect(redirect::Policy::limited(MAX_REDIRECTS))
            .build()
            .map_err(|e| LoadError::Network(e.to_string()))?;
        Ok(Self { client, max_bytes })
    }

    /// Build a loader with the default byte limit and timeout.
    pub fn with_defaults() -> Result<Self, LoadError> {
        Self::new(DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT)
    }

    /// Fetch `url` and return the response body.
    pub async fn load(&self, url: &str) -> Result<Vec<u8>, LoadError> {
        let parsed =
            Url::parse(url).map_err(|e| LoadError::Network(format!("invalid url: {e}")))?;
        let mut resp = self
            .client
            .get(parsed)
            .send()
            .await
            .map_err(map_request_error)?;
        let status = resp.status();
        if !status.is_success() {
            return Err(LoadError::BadStatus(status.as_u16()));
        }
        if let Some(value) = resp.headers().get(header::CONTENT_TYPE) {
            let content_type = value.to_str().ok();
            let is_image = content_type.map_or(false, |ct| ct.starts_with("image/"));
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
}
