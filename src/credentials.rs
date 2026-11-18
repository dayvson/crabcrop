//! Where the AWS signing credentials come from (issue #709).
//!
//! `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, with the
//! optional `AWS_SESSION_TOKEN`, are read once at startup and
//! used as they are, as before. When they are unset and the
//! environment names a container credentials endpoint, the
//! one ECS serves a task role on and EKS Pod Identity serves a
//! pod's role on, the credentials come from there instead, and
//! a background task fetches fresh ones
//! [`REFRESH_AHEAD`] before each set expires, so temporary
//! credentials no longer run out until a restart. One source
//! signs every AWS request the server makes: bucket source
//! fetches, uploads, and the shared result tier.
//!
//! The endpoint follows the AWS SDKs' container provider:
//! `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` is a path on ECS's
//! agent at `169.254.170.2`; otherwise
//! `AWS_CONTAINER_CREDENTIALS_FULL_URI` is the whole URL, over
//! HTTPS or, over plain HTTP, only to the ECS and EKS agents
//! or a loopback address. The request carries the token in
//! `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE` (read afresh for
//! every fetch, as EKS rotates it) or else
//! `AWS_CONTAINER_AUTHORIZATION_TOKEN`, as its
//! `Authorization` header. No other part of the AWS
//! credential chain is consulted: not web identity tokens
//! (IRSA), not EC2 instance metadata, not profile files.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::sigv4::Credentials;

/// Where ECS serves `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`.
pub const ECS_AGENT: &str = "http://169.254.170.2";

/// The hosts a full container credentials URI may name over
/// plain HTTP, besides loopback addresses: ECS's agent, and
/// EKS Pod Identity's on IPv4 and IPv6. Any other host must be
/// reached over HTTPS, as in the AWS SDKs: the request carries
/// the authorization token, and its answer is the credentials.
const PLAIN_HTTP_AGENTS: [&str; 3] = ["169.254.170.2", "169.254.170.23", "[fd00:ec2::23]"];

/// How long before a set of credentials expires the next one
/// is fetched. AWS's own SDKs refresh 5 to 15 minutes ahead;
/// the agents hand out a new set well before the old one
/// expires.
pub const REFRESH_AHEAD: Duration = Duration::from_secs(15 * 60);

/// The shortest wait before another fetch: the wait after a
/// fetch fails, and the floor for credentials that expire
/// soon.
const RETRY_AFTER: Duration = Duration::from_secs(30);

/// How often credentials whose answer names no expiry are
/// fetched again.
const NO_EXPIRY_REFRESH: Duration = Duration::from_secs(60 * 60);

/// The bounds on one fetch from the endpoint, a local agent.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// The most of an answer a fetch reads: credentials are a few
/// hundred bytes, and a larger answer is not one.
const MAX_ANSWER_BYTES: usize = 64 * 1024;

/// The signing credentials of the AWS requests: fixed, or
/// fetched and refreshed from the container endpoint. Clones
/// share one source, so every holder sees each refresh.
#[derive(Clone)]
pub struct AwsCredentials(Arc<Source>);

enum Source {
    Fixed(Arc<Credentials>),
    Container(Container),
}

struct Container {
    endpoint: ContainerEndpoint,
    client: reqwest::Client,
    /// The last credentials fetched, and when they expire.
    current: Mutex<Option<Fetched>>,
    refresh_ahead: Duration,
    /// The shortest wait before another fetch.
    floor: Duration,
    /// Whether the refresh task runs.
    started: AtomicBool,
}

#[derive(Clone)]
struct Fetched {
    creds: Arc<Credentials>,
    expires: Option<SystemTime>,
}

/// The container credentials endpoint the environment names,
/// and the token each fetch carries.
#[derive(Clone, PartialEq, Eq)]
pub struct ContainerEndpoint {
    pub url: String,
    pub token: Option<ContainerToken>,
    /// The variable the URL came from, for the boot line.
    pub var: &'static str,
}

/// Where a fetch's `Authorization` token comes from.
#[derive(Clone, PartialEq, Eq)]
pub enum ContainerToken {
    /// `AWS_CONTAINER_AUTHORIZATION_TOKEN`'s value, zeroed when
    /// it is dropped, like the other secrets.
    Value(zeroize::Zeroizing<String>),
    /// `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE`: read at every
    /// fetch.
    File(PathBuf),
}

impl std::fmt::Debug for ContainerToken {
    /// The token prints as `[redacted]`; a file prints its
    /// path, not what it holds.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerToken::Value(_) => f.write_str("Value([redacted])"),
            ContainerToken::File(path) => f.debug_tuple("File").field(path).finish(),
        }
    }
}

impl std::fmt::Debug for ContainerEndpoint {
    /// The URL prints without its path: ECS's relative URI
    /// names the task's credentials.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerEndpoint")
            .field("origin", &origin_of(&self.url))
            .field("token", &self.token)
            .field("var", &self.var)
            .finish()
    }
}

impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.0 {
            Source::Fixed(creds) => f.debug_tuple("Fixed").field(creds).finish(),
            Source::Container(container) => f
                .debug_struct("Container")
                .field("endpoint", &container.endpoint)
                .field(
                    "current",
                    &container.current().map(|creds| creds.access_key.clone()),
                )
                .finish(),
        }
    }
}

impl PartialEq for AwsCredentials {
    /// Two sources are equal when they are configured alike:
    /// the same fixed keys, or the same endpoint and token.
    /// What a container source has fetched does not count.
    fn eq(&self, other: &Self) -> bool {
        match (&*self.0, &*other.0) {
            (Source::Fixed(a), Source::Fixed(b)) => a == b,
            (Source::Container(a), Source::Container(b)) => a.endpoint == b.endpoint,
            _ => false,
        }
    }
}

impl Eq for AwsCredentials {}

impl From<Credentials> for AwsCredentials {
    fn from(creds: Credentials) -> Self {
        AwsCredentials(Arc::new(Source::Fixed(Arc::new(creds))))
    }
}

impl AwsCredentials {
    /// A source for the container credentials `endpoint`, with
    /// nothing fetched yet: [`AwsCredentials::start`] fetches.
    /// Fails when its HTTP client cannot be built.
    pub fn container(endpoint: ContainerEndpoint) -> Result<Self, reqwest::Error> {
        Self::container_refreshing(endpoint, REFRESH_AHEAD, RETRY_AFTER)
    }

    /// [`AwsCredentials::container`], refreshing `refresh_ahead`
    /// before each expiry and waiting at least `floor` between
    /// fetches: the tests shorten both.
    pub(crate) fn container_refreshing(
        endpoint: ContainerEndpoint,
        refresh_ahead: Duration,
        floor: Duration,
    ) -> Result<Self, reqwest::Error> {
        // Never through a proxy: the endpoint is the task's or
        // the pod's own agent, and its answer is credentials.
        // Never to a redirect's target either: the plain-HTTP
        // rule (see `container_endpoint`) holds for the URL the
        // environment names, and a redirect could lead the
        // fetch anywhere else, over anything.
        let client = crate::tls::client_builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(FETCH_TIMEOUT)
            .build()?;
        Ok(AwsCredentials(Arc::new(Source::Container(Container {
            endpoint,
            client,
            current: Mutex::new(None),
            refresh_ahead,
            floor,
            started: AtomicBool::new(false),
        }))))
    }

    /// The credentials to sign with now: the fixed ones, or the
    /// last set the endpoint gave, which a refresh replaces
    /// before it expires. `None` before the first fetch
    /// succeeds: the request then goes unsigned, and the store
    /// refuses it, which its tier counts and logs.
    pub fn current(&self) -> Option<Arc<Credentials>> {
        match &*self.0 {
            Source::Fixed(creds) => Some(creds.clone()),
            Source::Container(container) => container.current(),
        }
    }

    /// The fixed credentials, or `None` for a container source.
    pub fn fixed(&self) -> Option<&Credentials> {
        match &*self.0 {
            Source::Fixed(creds) => Some(creds),
            Source::Container(_) => None,
        }
    }

    /// Whether these come from the container endpoint.
    pub fn is_container(&self) -> bool {
        matches!(&*self.0, Source::Container(_))
    }

    /// Fetch the first credentials from the container endpoint
    /// and keep them fresh from then on, in a task of its own;
    /// a fixed source has nothing to do. Boot calls it once,
    /// before the first AWS request, and prints the line it
    /// returns. A first fetch that fails does not stop boot:
    /// the task retries every 30 seconds, and the AWS requests
    /// go unsigned until one succeeds.
    pub async fn start(&self) -> Option<String> {
        let Source::Container(container) = &*self.0 else {
            return None;
        };
        if container.started.swap(true, Ordering::SeqCst) {
            return None;
        }
        let first = container.refresh().await;
        let line = match &first {
            Ok(fetched) => format!(
                "aws credentials: from the container credentials endpoint ({}), {}",
                container.endpoint.var,
                match fetched.expires {
                    Some(at) => format!(
                        "expiring in {}s, refreshed {}s before",
                        seconds_until(at),
                        container.refresh_ahead.as_secs()
                    ),
                    None => format!(
                        "with no expiry, refreshed every {}s",
                        NO_EXPIRY_REFRESH.as_secs()
                    ),
                }
            ),
            Err(why) => format!(
                "warning: aws credentials: cannot fetch from the container credentials \
                 endpoint ({}): {why}; AWS requests go unsigned until a retry succeeds, \
                 every {}s",
                container.endpoint.var,
                container.floor.as_secs()
            ),
        };
        let mut wait = match &first {
            Ok(fetched) => container.next_wait(fetched),
            Err(_) => container.floor,
        };
        let source = self.clone();
        tokio::spawn(async move {
            let Source::Container(container) = &*source.0 else {
                return;
            };
            loop {
                tokio::time::sleep(wait).await;
                wait = match container.refresh().await {
                    Ok(fetched) => container.next_wait(&fetched),
                    Err(why) => {
                        eprintln!(
                            "aws credentials: cannot refresh from the container credentials \
                             endpoint: {why}; {}, retrying in {}s",
                            container.standing(),
                            container.floor.as_secs()
                        );
                        container.floor
                    }
                };
            }
        });
        Some(line)
    }
}

impl Container {
    /// How long to wait after a fetch that gave `fetched`: never
    /// less than the floor, so credentials that have expired, or
    /// nearly, do not make a busy loop.
    fn next_wait(&self, fetched: &Fetched) -> Duration {
        refresh_delay(SystemTime::now(), fetched.expires, self.refresh_ahead).max(self.floor)
    }

    fn current(&self) -> Option<Arc<Credentials>> {
        self.current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|fetched| fetched.creds.clone())
    }

    /// What the requests sign with while a refresh fails: the
    /// set still in force, one that has expired, or none.
    fn standing(&self) -> String {
        let current = self
            .current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match current {
            None => "AWS requests go unsigned".to_string(),
            Some(Fetched {
                expires: Some(at), ..
            }) if at <= SystemTime::now() => {
                "the last credentials have expired, so the bucket refuses AWS requests".to_string()
            }
            Some(Fetched {
                expires: Some(at), ..
            }) => format!("the last credentials expire in {}s", seconds_until(at)),
            Some(Fetched { expires: None, .. }) => "the last credentials stay in use".to_string(),
        }
    }

    /// One fetch; on success the answer replaces the current
    /// credentials.
    async fn refresh(&self) -> Result<Fetched, String> {
        let fetched = self.fetch().await?;
        *self
            .current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(fetched.clone());
        Ok(fetched)
    }

    /// Ask the endpoint for credentials. The error names what
    /// failed, never the token, the credentials, or the URL's
    /// path.
    async fn fetch(&self) -> Result<Fetched, String> {
        let mut request = self.client.get(&self.endpoint.url);
        if let Some(token) = &self.endpoint.token {
            request = request.header(reqwest::header::AUTHORIZATION, read_token(token)?.as_str());
        }
        let mut resp = request
            .send()
            .await
            .map_err(|e| format!("no answer ({})", e.without_url()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("the endpoint answered {status}"));
        }
        let mut body = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| format!("the answer broke off ({})", e.without_url()))?
        {
            if body.len() + chunk.len() > MAX_ANSWER_BYTES {
                return Err(format!("an answer past {MAX_ANSWER_BYTES} bytes"));
            }
            body.extend_from_slice(&chunk);
        }
        parse_answer(&body)
    }
}

/// The token a fetch sends, from the variable or the file,
/// with surrounding whitespace dropped (a file may end in a
/// newline). A token that could break out of its header is
/// refused. Each copy is zeroed when it is dropped, the
/// file's whole text included.
fn read_token(token: &ContainerToken) -> Result<zeroize::Zeroizing<String>, String> {
    let value = match token {
        ContainerToken::Value(value) => zeroize::Zeroizing::new(value.trim().to_string()),
        ContainerToken::File(path) => {
            let text = zeroize::Zeroizing::new(std::fs::read_to_string(path).map_err(|e| {
                format!(
                    "cannot read AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE {}: {e}",
                    path.display()
                )
            })?);
            zeroize::Zeroizing::new(text.trim().to_string())
        }
    };
    if value.is_empty() || !header_safe(&value) {
        return Err("the authorization token is empty or not a header value".to_string());
    }
    Ok(value)
}

/// Whether `value` can travel as a header value: printable
/// ASCII and spaces, no line breaks.
fn header_safe(value: &str) -> bool {
    value.bytes().all(|b| b == b' ' || b.is_ascii_graphic())
}

/// The credentials in an endpoint's JSON answer, which names
/// them as the AWS SDKs read them: `AccessKeyId`,
/// `SecretAccessKey`, `Token` and `Expiration`, an RFC 3339
/// time.
fn parse_answer(body: &[u8]) -> Result<Fetched, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct Answer {
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        token: Option<String>,
        expiration: Option<String>,
    }
    let answer: Answer = serde_json::from_slice(body)
        .map_err(|_| "an answer that is not the credentials JSON".to_string())?;
    let (Some(access_key), Some(secret_key)) = (answer.access_key_id, answer.secret_access_key)
    else {
        return Err("an answer with no AccessKeyId or SecretAccessKey".to_string());
    };
    let session_token = answer.token.filter(|t| !t.is_empty());
    if access_key.is_empty()
        || secret_key.is_empty()
        || !header_safe(&access_key)
        || !header_safe(&secret_key)
        || session_token.as_deref().is_some_and(|t| !header_safe(t))
    {
        return Err("an answer whose credentials are empty or not header values".to_string());
    }
    let expires = match answer.expiration {
        None => None,
        Some(text) => Some(
            parse_rfc3339(&text).ok_or_else(|| "an Expiration that is not a time".to_string())?,
        ),
    };
    Ok(Fetched {
        creds: Arc::new(Credentials {
            access_key,
            secret_key,
            session_token,
        }),
        expires,
    })
}

/// How long to wait before the next fetch, after one at `now`
/// whose credentials expire at `expires`: until `ahead` before
/// the expiry, or for credentials that expire sooner than
/// twice that, half of what they have left. The caller
/// applies its floor ([`RETRY_AFTER`] outside the tests).
fn refresh_delay(now: SystemTime, expires: Option<SystemTime>, ahead: Duration) -> Duration {
    let Some(expires) = expires else {
        return NO_EXPIRY_REFRESH;
    };
    let left = expires.duration_since(now).unwrap_or_default();
    if left > ahead * 2 {
        left - ahead
    } else {
        left / 2
    }
}

/// Whole seconds from now until `at`, 0 once past.
fn seconds_until(at: SystemTime) -> u64 {
    at.duration_since(SystemTime::now())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `scheme://host[:port]` of `url`, for printing without the
/// path.
fn origin_of(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(parsed) => parsed.origin().ascii_serialization(),
        Err(_) => "[not a URL]".to_string(),
    }
}

/// Parse an RFC 3339 time, `YYYY-MM-DDTHH:MM:SS`, optional
/// fractional seconds, then `Z` or a `+HH:MM` / `-HH:MM`
/// offset (the form the endpoints write, `2026-11-16T10:00:00Z`).
pub fn parse_rfc3339(text: &str) -> Option<SystemTime> {
    let b = text.as_bytes();
    let digits = |range: std::ops::Range<usize>| -> Option<i64> {
        let part = b.get(range)?;
        if !part.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(part).ok()?.parse().ok()
    };
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || !matches!(b[10], b'T' | b't')
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let (year, month, day) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
    let (hour, minute, second) = (digits(11..13)?, digits(14..16)?, digits(17..19)?);
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let mut rest = &text[19..];
    if let Some(fraction) = rest.strip_prefix('.') {
        let end = fraction
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(fraction.len());
        if end == 0 {
            return None;
        }
        rest = &fraction[end..];
    }
    let offset = match rest {
        "Z" | "z" => 0,
        _ => {
            let r = rest.as_bytes();
            if r.len() != 6 || !matches!(r[0], b'+' | b'-') || r[3] != b':' {
                return None;
            }
            let hours: i64 = rest[1..3].parse().ok()?;
            let minutes: i64 = rest[4..6].parse().ok()?;
            if hours > 23 || minutes > 59 {
                return None;
            }
            let span = hours * 3600 + minutes * 60;
            if r[0] == b'+' { span } else { -span }
        }
    };
    let secs =
        days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second.min(59)
            - offset;
    u64::try_from(secs)
        .ok()
        .map(|secs| UNIX_EPOCH + Duration::from_secs(secs))
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days from 1970-01-01 to the civil date (Howard Hinnant's
/// `days_from_civil`, the inverse of the conversion in
/// `sigv4::amz_date`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The container endpoint the environment names, or `None`
/// when it names none (issue #709): the relative URI on ECS's
/// agent first, as the AWS SDKs read it, then the full URI.
/// A full URI over plain HTTP to a host other than the agents
/// or a loopback address is refused, and so is a relative URI
/// that is not a path; the error names the variable and why.
pub fn container_endpoint(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<ContainerEndpoint>, (&'static str, String)> {
    let nonempty = |name: &str| get(name).filter(|v| !v.is_empty());
    let (url, var) = if let Some(relative) = nonempty("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
        if !relative.starts_with('/') {
            return Err((
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI must be a path, starting with /"
                    .to_string(),
            ));
        }
        (
            format!("{ECS_AGENT}{relative}"),
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        )
    } else if let Some(full) = nonempty("AWS_CONTAINER_CREDENTIALS_FULL_URI") {
        const VAR: &str = "AWS_CONTAINER_CREDENTIALS_FULL_URI";
        let parsed =
            reqwest::Url::parse(&full).map_err(|_| (VAR, format!("{VAR} is not a URL")))?;
        let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
        let plain_ok = PLAIN_HTTP_AGENTS.contains(&host.as_str())
            || host == "localhost"
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback());
        match parsed.scheme() {
            "https" => {}
            "http" if plain_ok => {}
            "http" => {
                return Err((
                    VAR,
                    format!(
                        "{VAR} over plain http must name the ECS or EKS agent \
                         (169.254.170.2, 169.254.170.23, fd00:ec2::23) or a loopback \
                         address; use https for any other host"
                    ),
                ));
            }
            _ => return Err((VAR, format!("{VAR} must be an http(s) URL"))),
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err((VAR, format!("{VAR} must not carry a user or password")));
        }
        (full, VAR)
    } else {
        return Ok(None);
    };
    let token = if let Some(path) = nonempty("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
        Some(ContainerToken::File(PathBuf::from(path)))
    } else if let Some(value) = nonempty("AWS_CONTAINER_AUTHORIZATION_TOKEN") {
        if !header_safe(value.trim()) {
            return Err((
                "AWS_CONTAINER_AUTHORIZATION_TOKEN",
                "AWS_CONTAINER_AUTHORIZATION_TOKEN must be one line of printable text".to_string(),
            ));
        }
        Some(ContainerToken::Value(value.into()))
    } else {
        None
    };
    Ok(Some(ContainerEndpoint { url, token, var }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| {
            pairs
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn the_container_token_zeroes_its_value() {
        // What dropping a token from the variable runs: its
        // bytes are wiped, as the other secrets' are.
        let endpoint = container_endpoint(env(&[
            (
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "http://127.0.0.1:8080/creds",
            ),
            ("AWS_CONTAINER_AUTHORIZATION_TOKEN", "pod-token"),
        ]))
        .unwrap()
        .unwrap();
        let Some(ContainerToken::Value(mut token)) = endpoint.token else {
            panic!("a token from AWS_CONTAINER_AUTHORIZATION_TOKEN");
        };
        assert_eq!(token.as_str(), "pod-token");
        zeroize::Zeroize::zeroize(&mut token);
        assert_eq!(token.as_str(), "");
        // The copy each fetch sends is zeroed when it is dropped
        // too, whether it came from the variable or the file.
        fn zeroed_when_dropped<T: zeroize::ZeroizeOnDrop>(_: &T) {}
        let sent = read_token(&ContainerToken::Value("pod-token".to_string().into())).unwrap();
        zeroed_when_dropped(&sent);
        assert_eq!(sent.as_str(), "pod-token");
    }

    #[test]
    fn the_endpoint_follows_the_sdk_precedence_and_host_rules() {
        assert_eq!(container_endpoint(env(&[])).unwrap(), None);
        // The relative URI is a path on ECS's agent, and wins
        // over a full URI.
        let ecs = container_endpoint(env(&[
            (
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/abc",
            ),
            (
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "https://elsewhere.example/c",
            ),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(ecs.url, "http://169.254.170.2/v2/credentials/abc");
        assert_eq!(ecs.token, None);
        assert!(
            container_endpoint(env(&[("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "v2/x")])).is_err()
        );
        // EKS Pod Identity: the agent over plain http, the token
        // in a file, which wins over the variable.
        let eks = container_endpoint(env(&[
            (
                "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "http://169.254.170.23/v1/credentials",
            ),
            ("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", "/var/run/token"),
            ("AWS_CONTAINER_AUTHORIZATION_TOKEN", "ignored"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            eks.token,
            Some(ContainerToken::File(PathBuf::from("/var/run/token")))
        );
        for allowed in [
            "http://[fd00:ec2::23]/v1/credentials",
            "http://127.0.0.1:8080/creds",
            "http://[::1]/creds",
            "http://localhost/creds",
            "https://credentials.example/creds",
        ] {
            assert!(
                container_endpoint(env(&[("AWS_CONTAINER_CREDENTIALS_FULL_URI", allowed)]))
                    .unwrap()
                    .is_some(),
                "{allowed}"
            );
        }
        // Plain http elsewhere would send the token, and take
        // credentials, in the clear.
        for refused in [
            "http://credentials.example/creds",
            "http://10.0.0.5/creds",
            "ftp://127.0.0.1/creds",
            "https://user:pass@credentials.example/creds",
            "not a url",
        ] {
            let err = container_endpoint(env(&[("AWS_CONTAINER_CREDENTIALS_FULL_URI", refused)]))
                .unwrap_err();
            assert_eq!(err.0, "AWS_CONTAINER_CREDENTIALS_FULL_URI", "{refused}");
        }
        let err = container_endpoint(env(&[
            (
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/abc",
            ),
            ("AWS_CONTAINER_AUTHORIZATION_TOKEN", "a\r\nX-Injected: 1"),
        ]))
        .unwrap_err();
        assert_eq!(err.0, "AWS_CONTAINER_AUTHORIZATION_TOKEN");
        assert!(!err.1.contains("Injected"), "{}", err.1);
    }

    #[test]
    fn expirations_parse_as_rfc3339() {
        let at = |secs: u64| Some(UNIX_EPOCH + Duration::from_secs(secs));
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), at(0));
        assert_eq!(parse_rfc3339("2015-08-30T12:36:00Z"), at(1_440_938_160));
        assert_eq!(parse_rfc3339("2020-02-29T00:00:00Z"), at(1_582_934_400));
        assert_eq!(parse_rfc3339("2020-12-31T23:59:59.999Z"), at(1_609_459_199));
        assert_eq!(
            parse_rfc3339("2021-01-01T02:00:00+02:00"),
            at(1_609_459_200)
        );
        assert_eq!(
            parse_rfc3339("2020-12-31T22:00:00-02:00"),
            at(1_609_459_200)
        );
        for bad in [
            "",
            "2021-01-01",
            "2021-01-01 00:00:00Z",
            "2021-13-01T00:00:00Z",
            "2021-02-29T00:00:00Z",
            "2021-01-01T24:00:00Z",
            "2021-01-01T00:00:00",
            "2021-01-01T00:00:00.Z",
            "2021-01-01T00:00:00+0200",
            "1969-12-31T23:59:59Z",
        ] {
            assert_eq!(parse_rfc3339(bad), None, "{bad}");
        }
    }

    #[test]
    fn refreshes_come_ahead_of_each_expiry() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000_000);
        let ahead = REFRESH_AHEAD;
        // Six hours left: a refresh 15 minutes before they end.
        let six_hours = now + Duration::from_secs(6 * 3600);
        assert_eq!(
            refresh_delay(now, Some(six_hours), ahead),
            Duration::from_secs(6 * 3600) - ahead
        );
        // Twenty minutes left: halfway, then sooner each time.
        assert_eq!(
            refresh_delay(now, Some(now + Duration::from_secs(1200)), ahead),
            Duration::from_secs(600)
        );
        // Expired: nothing left to wait for, so the source's
        // floor applies, not a busy loop.
        assert_eq!(refresh_delay(now, Some(now), ahead), Duration::ZERO);
        assert_eq!(
            refresh_delay(now, Some(now - Duration::from_secs(60)), ahead),
            Duration::ZERO
        );
        assert_eq!(refresh_delay(now, None, ahead), NO_EXPIRY_REFRESH);
        let source = AwsCredentials::container(ContainerEndpoint {
            url: "http://127.0.0.1:1/creds".to_string(),
            token: None,
            var: "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        })
        .unwrap();
        let Source::Container(container) = &*source.0 else {
            unreachable!("a container source")
        };
        let expired = Fetched {
            creds: Arc::new(Credentials {
                access_key: "A".to_string(),
                secret_key: "s".to_string(),
                session_token: None,
            }),
            expires: Some(SystemTime::now()),
        };
        assert_eq!(container.next_wait(&expired), RETRY_AFTER);
    }

    #[test]
    fn answers_parse_strictly_and_never_echo_secrets() {
        let good = br#"{"RoleArn":"arn:aws:iam::1:role/r","AccessKeyId":"ASIAEXAMPLE","SecretAccessKey":"secret/key+1","Token":"tok","Expiration":"2030-01-01T00:00:00Z"}"#;
        let fetched = parse_answer(good).unwrap();
        assert_eq!(fetched.creds.access_key, "ASIAEXAMPLE");
        assert_eq!(fetched.creds.secret_key, "secret/key+1");
        assert_eq!(fetched.creds.session_token.as_deref(), Some("tok"));
        assert_eq!(fetched.expires, parse_rfc3339("2030-01-01T00:00:00Z"));
        for bad in [
            &b"not json"[..],
            br#"{"AccessKeyId":"A"}"#,
            br#"{"AccessKeyId":"","SecretAccessKey":"s"}"#,
            br#"{"AccessKeyId":"A","SecretAccessKey":"s\nX: y"}"#,
            br#"{"AccessKeyId":"A","SecretAccessKey":"hunter2","Expiration":"soon"}"#,
        ] {
            let err = parse_answer(bad).err().unwrap();
            assert!(!err.contains("hunter2") && !err.contains("X: y"), "{err}");
        }
    }

    #[tokio::test]
    async fn a_redirect_is_not_followed() {
        // The host rules hold for the URL the environment
        // names: a redirect could send the fetch, token and
        // all, anywhere, and take credentials from there. The
        // endpoint's redirect is an error, and its target is
        // never asked.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let elsewhere = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target = format!("http://{}/creds", elsewhere.local_addr().unwrap());
        let reached = Arc::new(AtomicBool::new(false));
        let seen = reached.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = elsewhere.accept().await {
                seen.store(true, Ordering::SeqCst);
                let mut request = [0u8; 4096];
                let _ = socket.read(&mut request).await;
                let body = r#"{"AccessKeyId":"ASIAELSEWHERE","SecretAccessKey":"s"}"#;
                let answer = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(answer.as_bytes()).await;
            }
        });
        let endpoint = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/credentials", endpoint.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = endpoint.accept().await {
                let mut request = [0u8; 4096];
                let _ = socket.read(&mut request).await;
                let answer = format!(
                    "HTTP/1.1 302 Found\r\nlocation: {target}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                );
                let _ = socket.write_all(answer.as_bytes()).await;
            }
        });
        let source = AwsCredentials::container(ContainerEndpoint {
            url,
            token: Some(ContainerToken::Value("pod-token".to_string().into())),
            var: "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        })
        .unwrap();
        let Source::Container(container) = &*source.0 else {
            unreachable!("a container source")
        };
        let err = container.refresh().await.err().unwrap();
        assert!(err.contains("302"), "{err}");
        assert!(!reached.load(Ordering::SeqCst));
        assert_eq!(source.current(), None);
    }

    #[test]
    fn debug_prints_neither_token_nor_path_nor_secret() {
        let source = AwsCredentials::container(ContainerEndpoint {
            url: "http://169.254.170.2/v2/credentials/task-guid-1234".to_string(),
            token: Some(ContainerToken::Value(
                "bearer-secret-token".to_string().into(),
            )),
            var: "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        })
        .unwrap();
        let rendered = format!("{source:?}");
        assert!(rendered.contains("http://169.254.170.2"), "{rendered}");
        assert!(!rendered.contains("task-guid-1234"), "{rendered}");
        assert!(!rendered.contains("bearer-secret-token"), "{rendered}");
        let fixed: AwsCredentials = Credentials {
            access_key: "AKID".to_string(),
            secret_key: "fixed-secret".to_string(),
            session_token: None,
        }
        .into();
        assert!(!format!("{fixed:?}").contains("fixed-secret"));
        assert!(fixed.fixed().is_some() && !fixed.is_container());
        assert!(source.is_container() && source.fixed().is_none());
        assert_eq!(source.current(), None);
    }
}
