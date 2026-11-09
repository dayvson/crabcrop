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
//! request input. Object keys carrying dot segments are refused:
//! the request URL is path-style, and URL parsing would collapse
//! the segments past the allowlisted bucket after the check.
//! Store responses are not redirected; a 3xx from
//! a store surfaces as its status.
//!
//! A [`SourcePolicy`] can restrict which sources may be fetched: an
//! allowlist of host suffixes, and blocking of hosts that resolve to
//! loopback, private, link-local, or unspecified addresses. Both
//! checks run before any connection is made, and again on every
//! redirect hop (redirects are followed manually for exactly that
//! reason). When the private-address check resolves a hostname,
//! the connection is pinned to the validated addresses, so a
//! hostile DNS server cannot answer differently the second time
//! (DNS rebinding, audit L-3). Source URLs carrying userinfo are
//! refused: the loader never presents Basic credentials to an
//! origin (audit L-2).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::{Client, Url, header, redirect};

use crate::sigv4::{self, Credentials};

/// Default cap on response bodies: 10 MiB, the default of
/// `CRABCROP_SOURCE_MAX_MB` (issue #471). The upload
/// endpoint's size cap is tied to the configured cap
/// (issue #168): an original larger than the loader will
/// fetch could be stored but never served.
pub const DEFAULT_MAX_BYTES: u64 = 10 * 1024 * 1024;
/// Default total request timeout: 10 seconds, the default
/// of `CRABCROP_SOURCE_TIMEOUT_SECS` (issue #471).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
/// Maximum redirects followed per request.
const MAX_REDIRECTS: usize = 5;

/// What went wrong while fetching a source image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// DNS, connect, TLS, or body-stream failure. Carries the cause.
    Network(String),
    /// The source text the request named is not one the
    /// loader can fetch: it does not parse, names a scheme
    /// the loader does not serve, or is a malformed bucket
    /// or file locator. The fault is the caller's own URL,
    /// so it is not a gateway failure (issue #467). Only
    /// the first hop: a redirect to such a URL is the
    /// origin's answer and stays [`LoadError::Network`].
    /// Carries the cause.
    InvalidSource(String),
    /// The source does not exist: a file missing under a file
    /// root, a path that resolves outside every root, or an
    /// origin (an object store included) that answered 404.
    /// One variant for all three, so the answer is never an
    /// oracle for what exists (issue #325).
    NotFound,
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
    /// The source buffer budget (issue #403) stayed full for
    /// the whole wait the request had for it: the load was
    /// abandoned before it buffered the body, or the rest of
    /// it.
    Busy,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Network(cause) => write!(f, "network error: {cause}"),
            LoadError::InvalidSource(cause) => write!(f, "invalid source: {cause}"),
            LoadError::NotFound => write!(f, "source not found"),
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
            LoadError::Busy => write!(f, "source buffer budget exhausted"),
            LoadError::Forbidden(reason) => {
                write!(f, "source refused by policy: {reason}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// The non-image source types one fetch may admit, on
/// top of the image-only rule every fetch keeps. Each
/// admission is a property of the request, never of the
/// response alone: a `video/*` Content-Type is accepted
/// only for a request whose operations carry a `frame:`
/// segment (issue #214), and an `application/pdf`
/// Content-Type only for one carrying a `page:` segment
/// (issue #215). Content type by itself admits nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaAdmission {
    /// Accept `video/*` (a frame request).
    pub video: bool,
    /// Accept `application/pdf` (a page request).
    pub pdf: bool,
}

impl MediaAdmission {
    /// Whether this response's Content-Type is admitted:
    /// `image/*` always, the rest per the flags.
    fn accepts(&self, content_type: &str) -> bool {
        content_type.starts_with("image/")
            || (self.video && content_type.starts_with("video/"))
            || (self.pdf
                && content_type
                    .split(';')
                    .next()
                    .is_some_and(|ct| ct.trim() == "application/pdf"))
    }
}

/// The process-wide budget for source bytes held in memory
/// (issue #403). Cache misses charge their source and
/// watermark bodies against it as the bytes arrive (see
/// [`SourceCharge`]) and keep the charge until the render
/// job that uses the bytes finishes, so peak buffered
/// source memory is bounded by the budget rather than by
/// how many requests arrive at once. Clones share one
/// budget.
#[derive(Debug, Clone)]
pub struct SourceBudget {
    sem: std::sync::Arc<tokio::sync::Semaphore>,
    capacity: usize,
}

impl SourceBudget {
    /// A budget of `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.min(tokio::sync::Semaphore::MAX_PERMITS);
        Self {
            sem: std::sync::Arc::new(tokio::sync::Semaphore::new(capacity)),
            capacity,
        }
    }

    /// A budget nothing realistic exhausts, for loaders built
    /// by tests and tools rather than the server.
    pub fn unlimited() -> Self {
        Self::new(tokio::sync::Semaphore::MAX_PERMITS)
    }

    /// The budget, in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently charged against the budget.
    pub fn in_use(&self) -> usize {
        self.capacity.saturating_sub(self.sem.available_permits())
    }

    /// Open an empty charge for one request, which may wait
    /// at most `wait` in total for budget across all its
    /// loads.
    pub fn charge(&self, wait: Duration) -> SourceCharge {
        SourceCharge {
            budget: self.clone(),
            state: std::sync::Mutex::new(ChargeState {
                permit: None,
                wait_left: wait,
            }),
        }
    }
}

/// One request's charge against a [`SourceBudget`] (issue
/// #403): the bytes its loads hold, and the wait it has
/// left for more. A charged load reserves nothing before
/// the origin answers. It then holds what its body has
/// received plus at most [`BODY_READ_AHEAD`], and the rest
/// of its length only once it has received the read-ahead,
/// and it keeps budget past its received bytes only while
/// the body arrives fast enough (see [`BODY_GRANT`]). A
/// body that falls behind, stalled or trickling, holds
/// only the bytes it has delivered, never budget it is not
/// filling. Dropping the charge returns its bytes to the
/// budget.
#[derive(Debug)]
pub struct SourceCharge {
    budget: SourceBudget,
    state: std::sync::Mutex<ChargeState>,
}

#[derive(Debug)]
struct ChargeState {
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
    wait_left: Duration,
}

impl SourceCharge {
    /// The bytes this charge holds.
    pub fn bytes(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.permit.as_ref().map_or(0, |p| p.num_permits())
    }

    /// Grow the charge to `total` bytes, waiting for the
    /// difference at most the wait this request has left. A
    /// total past the whole budget could never be granted
    /// and fails at once. [`LoadError::Busy`] when the
    /// budget stayed exhausted.
    pub async fn grow_to(&self, total: usize) -> Result<(), LoadError> {
        let need = {
            let state = self.state.lock().unwrap();
            let held = state.permit.as_ref().map_or(0, |p| p.num_permits());
            if total <= held {
                return Ok(());
            }
            total - held
        };
        if total > self.budget.capacity {
            return Err(LoadError::Busy);
        }
        let permits = u32::try_from(need).map_err(|_| LoadError::Busy)?;
        let permit = self
            .acquire(self.budget.sem.clone(), permits)
            .await
            .ok_or(LoadError::Busy)?;
        let mut state = self.state.lock().unwrap();
        match &mut state.permit {
            Some(held) => held.merge(permit),
            None => state.permit = Some(permit),
        }
        Ok(())
    }

    /// Take `permits` from `sem`, waiting at most the wait
    /// this request has left and taking the time spent out
    /// of it. `None` when the wait ran out.
    async fn acquire(
        &self,
        sem: std::sync::Arc<tokio::sync::Semaphore>,
        permits: u32,
    ) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let wait = self.state.lock().unwrap().wait_left;
        let start = tokio::time::Instant::now();
        let acquired = tokio::time::timeout(wait, sem.acquire_many_owned(permits)).await;
        let mut state = self.state.lock().unwrap();
        state.wait_left = state.wait_left.saturating_sub(start.elapsed());
        acquired.ok()?.ok()
    }

    /// Keep `total` bytes of the charge and return the rest
    /// to the budget at once. Never grows the charge.
    pub fn shrink_to(&self, total: usize) {
        let mut state = self.state.lock().unwrap();
        if let Some(permit) = &mut state.permit {
            let excess = permit.num_permits().saturating_sub(total);
            if excess > 0 {
                drop(permit.split(excess));
            }
        }
    }

    /// Move `bytes` of this charge (all of it, when it holds
    /// fewer) into a new charge, so the two parts can be
    /// released separately. The new charge has no wait.
    pub fn split_off(&self, bytes: usize) -> SourceCharge {
        let mut state = self.state.lock().unwrap();
        let held = state.permit.as_ref().map_or(0, |p| p.num_permits());
        let permit = if bytes >= held {
            state.permit.take()
        } else {
            state.permit.as_mut().and_then(|p| p.split(bytes))
        };
        SourceCharge {
            budget: self.budget.clone(),
            state: std::sync::Mutex::new(ChargeState {
                permit,
                wait_left: Duration::ZERO,
            }),
        }
    }
}

/// The window over which a charged origin body's arrival
/// is judged. A body holds budget past the bytes it has
/// received (its read-ahead, then the rest of its length)
/// only while each window brings at least
/// [`MIN_BODY_RATE`], and enough that the body, arriving
/// at the window's rate, would fill what it holds before
/// the load times out. A body that falls short, an origin
/// that stalls or sends a few bytes at a time, drops to
/// the bytes it has received, and from then on each chunk
/// is charged as it arrives: budget is not held for a body
/// that would fill it slowly, or not before its timeout
/// cancels the load. A body that keeps up keeps its grant
/// for as long as it takes, so under a burst a healthy
/// load that is already arriving is not shed half-read.
pub const BODY_GRANT: Duration = Duration::from_secs(1);

/// The most a charged origin body reserves before any of
/// it has arrived: 1 MiB, or its declared length when that
/// is smaller. The body holds this read-ahead past its
/// received bytes until it has received that much, and
/// only then, if it is keeping up (see [`BODY_GRANT`]),
/// reserves the rest of its length (the fetch cap when it
/// declares none) in one step.
pub const BODY_READ_AHEAD: usize = 1024 * 1024;

/// The slowest arrival, in bytes per second, at which a
/// charged origin body keeps budget past the bytes it has
/// received (see [`BODY_GRANT`]): 64 KiB/s.
pub const MIN_BODY_RATE: usize = 64 * 1024;

/// The read-ahead pool for a budget of `capacity` bytes and
/// a fetch cap of `cap` (see `Loader::read_ahead_pool`):
/// the budget less the most a body can still need past its
/// read-ahead, and room for at least one read-ahead.
fn read_ahead_pool(capacity: usize, cap: u64) -> std::sync::Arc<tokio::sync::Semaphore> {
    let rest = usize::try_from(cap)
        .unwrap_or(usize::MAX)
        .saturating_sub(BODY_READ_AHEAD);
    let size = capacity
        .saturating_sub(rest)
        .clamp(BODY_READ_AHEAD, tokio::sync::Semaphore::MAX_PERMITS);
    std::sync::Arc::new(tokio::sync::Semaphore::new(size))
}

/// One body read's part of a request's [`SourceCharge`].
/// An origin body (see [`BodyCharge::stream`]) reserves
/// its read-ahead, then the rest of its length once it has
/// received that much, and holds budget past its received
/// bytes only while it keeps up (see [`BODY_GRANT`]). A
/// local file reserves its size up front: a local read
/// does not stall. Dropped
/// before [`BodyCharge::finish`] (an error, or the load's
/// timeout cancelling the read), it returns the part to
/// the budget, so a failed load leaves the request holding
/// only the bodies it already kept.
struct BodyCharge<'a> {
    charge: &'a SourceCharge,
    /// What the request held before this body.
    base: usize,
    /// The body's length as far as it is known before it
    /// arrives: its declared length, or the fetch cap when
    /// it declares none.
    length: usize,
    /// What this body may hold before it must take more
    /// budget: its read-ahead, then its length, while it
    /// keeps up; what it has received once it falls behind.
    granted: usize,
    /// The window being judged, while the body keeps budget
    /// past its received bytes. `None` for a file, and for a
    /// body that has fallen behind.
    window: Option<Window>,
    /// The body's place in the loader's read-ahead pool,
    /// held while its grant is a read-ahead short of its
    /// length.
    slot: Option<tokio::sync::OwnedSemaphorePermit>,
    finished: bool,
}

/// One window of a charged body's arrival (see
/// [`BODY_GRANT`]).
#[derive(Debug, Clone, Copy)]
struct Window {
    /// When the window opened.
    opened: tokio::time::Instant,
    /// What the body had received when it opened.
    received: usize,
    /// When the load times out.
    deadline: tokio::time::Instant,
}

impl<'a> BodyCharge<'a> {
    fn new(charge: &'a SourceCharge, length: usize) -> Self {
        BodyCharge {
            charge,
            base: charge.bytes(),
            length,
            granted: 0,
            window: None,
            slot: None,
            finished: false,
        }
    }

    /// Start charging an origin body (HTTP or an object
    /// store) once the origin has answered: a body that may
    /// outgrow its read-ahead first takes its place in
    /// `pool`, then the read-ahead is reserved and the first
    /// window opens. `deadline` is when the load times out.
    async fn stream(
        charge: &'a SourceCharge,
        declared: Option<u64>,
        cap: u64,
        pool: &std::sync::Arc<tokio::sync::Semaphore>,
        deadline: tokio::time::Instant,
    ) -> Result<BodyCharge<'a>, LoadError> {
        let cap = usize::try_from(cap).unwrap_or(usize::MAX);
        let length = declared
            .and_then(|d| usize::try_from(d).ok())
            .map_or(cap, |d| d.min(cap));
        let ahead = length.min(BODY_READ_AHEAD);
        let mut part = BodyCharge::new(charge, length);
        if ahead < length {
            let permits = u32::try_from(ahead).map_err(|_| LoadError::Busy)?;
            let slot = charge.acquire(pool.clone(), permits).await;
            part.slot = Some(slot.ok_or(LoadError::Busy)?);
        }
        charge.grow_to(part.base.saturating_add(ahead)).await?;
        part.granted = ahead;
        part.window = Some(Window {
            opened: tokio::time::Instant::now(),
            received: 0,
            deadline,
        });
        Ok(part)
    }

    /// Start charging a local file of `total` bytes, its
    /// size reserved before any of it is read.
    async fn file(charge: &'a SourceCharge, total: usize) -> Result<BodyCharge<'a>, LoadError> {
        let mut part = BodyCharge::new(charge, total);
        charge.grow_to(part.base.saturating_add(total)).await?;
        part.granted = total;
        Ok(part)
    }

    /// When the window being judged closes, while the body
    /// keeps budget past its received bytes.
    fn window_closes(&self) -> Option<tokio::time::Instant> {
        self.window.map(|w| w.opened + BODY_GRANT)
    }

    /// Judge the window once it has closed, the body having
    /// received `received` bytes: a body that kept up opens
    /// the next window; one that fell behind ends its grant
    /// and from here holds what it has received.
    fn judge(&mut self, received: usize) {
        let Some(window) = self.window else {
            return;
        };
        let now = tokio::time::Instant::now();
        let elapsed = now.saturating_duration_since(window.opened);
        if elapsed < BODY_GRANT {
            return;
        }
        let elapsed = elapsed.as_millis();
        let arrived = received.saturating_sub(window.received) as u128;
        let ahead = self.granted.saturating_sub(received) as u128;
        let left = window.deadline.saturating_duration_since(now).as_millis();
        let kept_up =
            arrived * 1000 >= MIN_BODY_RATE as u128 * elapsed && arrived * left >= ahead * elapsed;
        if kept_up {
            self.window = Some(Window {
                opened: now,
                received,
                ..window
            });
        } else {
            self.window = None;
            self.slot = None;
            self.granted = received;
            self.charge.shrink_to(self.base + received);
        }
    }

    /// Make sure `received` bytes are charged before the
    /// body holds them: free within the grant. A body that
    /// has received its read-ahead while keeping up reserves
    /// the rest of its length in one step and leaves the
    /// read-ahead pool; any other body grows by exactly what
    /// arrived.
    async fn cover(&mut self, received: usize) -> Result<(), LoadError> {
        if received <= self.granted {
            return Ok(());
        }
        if let Some(window) = self.window
            && received <= self.length
        {
            self.charge.grow_to(self.base + self.length).await?;
            self.granted = self.length;
            self.slot = None;
            // Nothing arrived while the body waited for
            // budget, so the wait is not held against it.
            self.window = Some(Window {
                opened: tokio::time::Instant::now(),
                received,
                ..window
            });
            return Ok(());
        }
        self.charge.grow_to(self.base + received).await?;
        self.granted = received;
        Ok(())
    }

    /// Keep exactly the `received` bytes of a complete body.
    fn finish(mut self, received: usize) {
        self.charge.shrink_to(self.base + received);
        self.finished = true;
    }
}

impl Drop for BodyCharge<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.charge.shrink_to(self.base);
        }
    }
}

/// Which sources the loader may fetch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourcePolicy {
    /// Host suffixes an URL's host must match (exactly, or as a
    /// subdomain). Empty means no allowlist. A listed host is
    /// exempt from the private-address check: the allowlist is
    /// how an operator vouches for a loopback or internal
    /// source.
    pub allowed_suffixes: Vec<String>,
    /// Refuse hosts resolving to non-public addresses, except
    /// allowlisted hosts.
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

    /// Whether fetches under this policy pin hostnames to
    /// their validated addresses (see `check_policy`): the
    /// private-address check is on and no allowlist is set.
    /// With an allowlist, every host the loader fetches is
    /// listed, and listed hosts skip the check and the pin.
    /// A pinned fetch connects directly, never through a
    /// proxy the environment configures, which boot warns
    /// about (issue #471).
    pub fn pins_hostnames(&self) -> bool {
        self.block_private && self.allowed_suffixes.is_empty()
    }
}

/// Fetches source image bytes over HTTP(S).
#[derive(Debug, Clone)]
pub struct Loader {
    client: Client,
    timeout: Duration,
    max_bytes: u64,
    policy: SourcePolicy,
    sources: SourceConfig,
    /// Pinned clients, one per validated (host, address)
    /// pair (issue #369): a pinned client forces its
    /// host's DNS to the validated address, so clients
    /// cannot be shared across pairs, but a redirect chain
    /// (or the next fetch) that lands on a pair already
    /// seen reuses its client, connection pool and TLS
    /// configuration included, instead of rebuilding all
    /// of it per hop. The map is capped; past the cap it
    /// clears and rebuilds, which costs a rebuild, never
    /// correctness. Behind an Arc so loader clones share
    /// the cache.
    pinned_clients: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<(String, std::net::SocketAddr), Client>>,
    >,
    /// The source buffer budget (issue #403) the server's
    /// misses charge their loads against: see
    /// [`Loader::load_media_charged`]. Unlimited until the
    /// server installs the configured budget.
    source_budget: SourceBudget,
    /// The read-ahead pool: what bodies whose grant is still
    /// a read-ahead short of their length may reserve
    /// together (see [`BodyCharge`]). It is the budget less
    /// what one body can still need past its read-ahead, so
    /// read-aheads alone cannot take the budget the rest of
    /// a body that is keeping up needs: loads already
    /// arriving finish one after another instead of all
    /// waiting on each other until they shed. Other holders
    /// (bodies at their full length, small bodies, uploads)
    /// still compete for that room. Rebuilt with
    /// the budget; shared by loader clones.
    read_ahead_pool: std::sync::Arc<tokio::sync::Semaphore>,
}

/// Cap on the pinned-client cache (issue #369).
const MAX_PINNED_CLIENTS: usize = 64;

/// A hostname resolution the policy has already validated:
/// the connection must go to one of these addresses (L-3).
#[derive(Debug, Clone)]
struct PinnedResolution {
    host: String,
    addr: std::net::SocketAddr,
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
    /// `*` allows every bucket. An empty list refuses every
    /// bucket at fetch time, with or without credentials;
    /// boot additionally refuses the combination of
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

/// How much of a body a probe load reads at most: enough
/// for a header probe to identify any source the pipeline
/// reads, with room to spare for header-rich formats, and
/// no more.
const PROBE_HEAD_BYTES: usize = 256 * 1024;

/// What a probe load learned about a source: the head of
/// the body, and the body's total size when it is known
/// without reading the rest (a declared Content-Length, a
/// file's size, or a body that ended inside the head).
/// `None` means the size is genuinely unknown, never a
/// partial count dressed as a total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeLoad {
    pub head: Vec<u8>,
    pub total_bytes: Option<u64>,
}

/// Which body read a load performs: the full body, or a
/// probe that stops as soon as the caller's predicate
/// settles on the head (or the head cap is reached).
enum BodyMode<'a> {
    /// The whole body, charged against the request's
    /// source budget share when there is one (issue #403).
    Full(Option<Charged<'a>>),
    Probe(&'a (dyn Fn(&[u8]) -> bool + Sync)),
}

/// A charged full load: the request's share of the source
/// budget, and when the load times out, which a body's
/// arrival is judged against (see [`BODY_GRANT`]).
#[derive(Clone, Copy)]
struct Charged<'a> {
    charge: &'a SourceCharge,
    deadline: tokio::time::Instant,
}

/// A load's result, in the shape its mode asked for.
enum LoadBody {
    Full(Vec<u8>),
    Probe(ProbeLoad),
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
            timeout,
            max_bytes,
            policy: SourcePolicy::default(),
            sources: SourceConfig::default(),
            pinned_clients: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            source_budget: SourceBudget::unlimited(),
            read_ahead_pool: read_ahead_pool(tokio::sync::Semaphore::MAX_PERMITS, max_bytes),
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

    /// Set the source buffer budget (issue #403).
    pub fn with_source_budget(mut self, budget: SourceBudget) -> Self {
        self.read_ahead_pool = read_ahead_pool(budget.capacity(), self.max_bytes);
        self.source_budget = budget;
        self
    }

    /// The source buffer budget this loader's callers
    /// charge against.
    pub fn source_budget(&self) -> &SourceBudget {
        &self.source_budget
    }

    /// Fetch `url` and return the body bytes, dispatching on
    /// the source scheme.
    pub async fn load(&self, url: &str) -> Result<Vec<u8>, LoadError> {
        self.load_media(url, MediaAdmission::default()).await
    }

    /// Fetch `url` like [`Loader::load`], with the
    /// non-image source types this request may admit (see
    /// [`MediaAdmission`]). All other guards (allowlist,
    /// dot segments, size cap, redirects, SSRF checks) are
    /// identical either way, and file sources never had a
    /// Content-Type to check. The loader's timeout is one
    /// total budget for the whole load: it spans every
    /// redirect hop and the body read, rather than
    /// restarting per request the way the HTTP client's own
    /// per-request timeout does, so a chain of individually
    /// fast hops cannot stretch a fetch past the total.
    pub async fn load_media(
        &self,
        url: &str,
        admission: MediaAdmission,
    ) -> Result<Vec<u8>, LoadError> {
        self.load_full(url, admission, None).await
    }

    /// [`Loader::load_media`], charging the body against
    /// `charge` as it arrives (issue #403): nothing before
    /// the origin answers, then what has been received plus
    /// a bounded read-ahead, and past the received bytes
    /// only while the body keeps up (see [`BODY_GRANT`] and
    /// [`BODY_READ_AHEAD`]). The charge keeps the body's
    /// bytes when the load succeeds and drops them when it
    /// fails. [`LoadError::Busy`] when the request's wait
    /// for budget ran out.
    pub async fn load_media_charged(
        &self,
        url: &str,
        admission: MediaAdmission,
        charge: &SourceCharge,
    ) -> Result<Vec<u8>, LoadError> {
        self.load_full(url, admission, Some(charge)).await
    }

    async fn load_full(
        &self,
        url: &str,
        admission: MediaAdmission,
        charge: Option<&SourceCharge>,
    ) -> Result<Vec<u8>, LoadError> {
        let deadline = tokio::time::Instant::now() + self.timeout;
        let charged = charge.map(|charge| Charged { charge, deadline });
        match self
            .load_timed(url, admission, BodyMode::Full(charged), deadline)
            .await?
        {
            LoadBody::Full(bytes) => Ok(bytes),
            LoadBody::Probe(_) => Err(LoadError::Network(
                "internal: probe result for a full load".to_string(),
            )),
        }
    }

    /// Fetch only what a probe of the source needs: the
    /// head of the body, read incrementally and stopped as
    /// soon as `settled` accepts the head (or the probe
    /// head cap is reached), under the same guards and the
    /// same total timeout as a full load. For a caller
    /// that only reads the source's headers (the explain
    /// report), so a slow or large origin cannot make the
    /// probe download a whole body it will never use.
    pub async fn load_probe(
        &self,
        url: &str,
        admission: MediaAdmission,
        settled: &(dyn Fn(&[u8]) -> bool + Sync),
    ) -> Result<ProbeLoad, LoadError> {
        let deadline = tokio::time::Instant::now() + self.timeout;
        match self
            .load_timed(url, admission, BodyMode::Probe(settled), deadline)
            .await?
        {
            LoadBody::Probe(probe) => Ok(probe),
            LoadBody::Full(_) => Err(LoadError::Network(
                "internal: full result for a probe load".to_string(),
            )),
        }
    }

    /// Run one load under the loader's total timeout, which
    /// ends at `deadline`.
    async fn load_timed(
        &self,
        url: &str,
        admission: MediaAdmission,
        mode: BodyMode<'_>,
        deadline: tokio::time::Instant,
    ) -> Result<LoadBody, LoadError> {
        match tokio::time::timeout_at(deadline, self.load_dispatch(url, admission, mode)).await {
            Ok(result) => result,
            Err(_) => Err(LoadError::Timeout),
        }
    }

    /// The scheme dispatch behind the public loads, run
    /// under the caller's total timeout.
    async fn load_dispatch(
        &self,
        url: &str,
        admission: MediaAdmission,
        mode: BodyMode<'_>,
    ) -> Result<LoadBody, LoadError> {
        if let Some(rest) = url.strip_prefix("s3://") {
            let (bucket, key) = split_bucket_key(rest, "s3")?;
            return self
                .load_object(StoreKind::S3, bucket, key, admission, mode)
                .await;
        }
        if let Some(rest) = url.strip_prefix("gs://") {
            let (bucket, key) = split_bucket_key(rest, "gs")?;
            return self
                .load_object(StoreKind::Gcs, bucket, key, admission, mode)
                .await;
        }
        if url.starts_with("file://") {
            return self.load_file(url, mode).await;
        }
        self.load_http(url, admission, mode).await
    }

    async fn load_http(
        &self,
        url: &str,
        admission: MediaAdmission,
        mode: BodyMode<'_>,
    ) -> Result<LoadBody, LoadError> {
        let mut parsed =
            Url::parse(url).map_err(|e| LoadError::InvalidSource(format!("invalid url: {e}")))?;
        let mut hops = 0;
        loop {
            if parsed.scheme() != "http" && parsed.scheme() != "https" {
                let cause = format!("unsupported scheme: {}", parsed.scheme());
                // The first hop is the caller's own text; a
                // redirect elsewhere is the origin's answer.
                return Err(if hops == 0 {
                    LoadError::InvalidSource(cause)
                } else {
                    LoadError::Network(cause)
                });
            }
            let pin = self.check_policy(&parsed).await?;
            let client = match &pin {
                Some(pin) => self.pinned_client(pin)?,
                None => self.client.clone(),
            };
            let resp = client
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
            return self.read_body(resp, admission, &mode).await;
        }
    }

    /// Read a response body in the shape the load's mode
    /// asked for (see [`BodyMode`]).
    async fn read_body(
        &self,
        resp: reqwest::Response,
        admission: MediaAdmission,
        mode: &BodyMode<'_>,
    ) -> Result<LoadBody, LoadError> {
        match mode {
            BodyMode::Full(charged) => Ok(LoadBody::Full(
                self.read_checked_body(resp, admission, *charged).await?,
            )),
            BodyMode::Probe(settled) => Ok(LoadBody::Probe(
                self.read_probe_body(resp, admission, settled).await?,
            )),
        }
    }

    /// The status and Content-Type checks both body reads
    /// share. A non-2xx status (including a store redirect,
    /// which is not followed) is an error. The Content-Type
    /// must be `image/*`, or one of the non-image types this
    /// request admits (see [`MediaAdmission`]). Returns the
    /// declared body length when the response carries one.
    fn check_response_head(
        resp: &reqwest::Response,
        admission: MediaAdmission,
    ) -> Result<Option<u64>, LoadError> {
        let status = resp.status();
        if status.as_u16() == 404 {
            // A missing object is not a gateway fault, and it
            // gets the same not-found answer as a missing file
            // (issue #325).
            return Err(LoadError::NotFound);
        }
        if !status.is_success() {
            return Err(LoadError::BadStatus(status.as_u16()));
        }
        if let Some(value) = resp.headers().get(header::CONTENT_TYPE) {
            let content_type = value.to_str().ok();
            let accepted = content_type.is_some_and(|ct| admission.accepts(ct));
            if !accepted {
                return Err(LoadError::WrongContentType(
                    content_type.map(str::to_string),
                ));
            }
        }
        Ok(resp.content_length())
    }

    /// Status, content-type, and bounded-body handling shared by
    /// the HTTP path and the object stores.
    async fn read_checked_body(
        &self,
        mut resp: reqwest::Response,
        admission: MediaAdmission,
        charged: Option<Charged<'_>>,
    ) -> Result<Vec<u8>, LoadError> {
        let declared = Self::check_response_head(&resp, admission)?;
        let Some(charged) = charged else {
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
        };
        // A charged read (issue #403) refuses a declared
        // length past the cap before it reserves anything,
        // then charges the body as described on
        // `BodyCharge`.
        if declared.is_some_and(|total| total > self.max_bytes) {
            return Err(LoadError::TooLarge {
                limit: self.max_bytes,
            });
        }
        let mut part = BodyCharge::stream(
            charged.charge,
            declared,
            self.max_bytes,
            &self.read_ahead_pool,
            charged.deadline,
        )
        .await?;
        let mut body = Vec::new();
        loop {
            part.judge(body.len());
            let next = match part.window_closes() {
                Some(closes) => match tokio::time::timeout_at(closes, resp.chunk()).await {
                    Ok(next) => next,
                    // The window closed first: judge it.
                    Err(_) => continue,
                },
                None => resp.chunk().await,
            };
            let Some(bytes) = next.map_err(map_request_error)? else {
                break;
            };
            let received = body.len() + bytes.len();
            if received as u64 > self.max_bytes {
                return Err(LoadError::TooLarge {
                    limit: self.max_bytes,
                });
            }
            part.cover(received).await?;
            body.extend_from_slice(&bytes);
        }
        part.finish(body.len());
        Ok(body)
    }

    /// The probe body read: the same head checks as a full
    /// load, then just enough of the body for the caller's
    /// probe to settle, under the same size cap. A declared
    /// length past the cap keeps the full load's too-large
    /// answer, so the probe never reports facts about a
    /// source a real request would refuse for its size.
    async fn read_probe_body(
        &self,
        mut resp: reqwest::Response,
        admission: MediaAdmission,
        settled: &(dyn Fn(&[u8]) -> bool + Sync),
    ) -> Result<ProbeLoad, LoadError> {
        let declared = Self::check_response_head(&resp, admission)?;
        if declared.is_some_and(|total| total > self.max_bytes) {
            return Err(LoadError::TooLarge {
                limit: self.max_bytes,
            });
        }
        let mut head = Vec::new();
        let mut received: u64 = 0;
        let mut ended = false;
        loop {
            if settled(&head) || head.len() >= PROBE_HEAD_BYTES {
                break;
            }
            let chunk = resp.chunk().await.map_err(map_request_error)?;
            let Some(bytes) = chunk else {
                ended = true;
                break;
            };
            received += bytes.len() as u64;
            if received > self.max_bytes {
                return Err(LoadError::TooLarge {
                    limit: self.max_bytes,
                });
            }
            let room = PROBE_HEAD_BYTES - head.len();
            head.extend_from_slice(&bytes[..bytes.len().min(room)]);
        }
        let total_bytes = declared.or_else(|| ended.then_some(received));
        Ok(ProbeLoad { head, total_bytes })
    }

    /// Fetch one object from S3 or GCS, signing with SigV4 when
    /// credentials are configured for the store.
    async fn load_object(
        &self,
        kind: StoreKind,
        bucket: &str,
        key: &str,
        admission: MediaAdmission,
        mode: BodyMode<'_>,
    ) -> Result<LoadBody, LoadError> {
        // The request URL places the key after the bucket, and
        // URL parsing collapses dot segments, so a key carrying
        // them would move the request outside the allowlisted
        // bucket after the check below, and the traversed
        // request would be validly signed (audit R2-05). Refuse
        // before anything is signed.
        if has_dot_segment(bucket) || has_dot_segment(key) {
            return Err(LoadError::Forbidden(
                "bucket or object key contains a dot segment".to_string(),
            ));
        }
        // Bucket sources are opt-in: the bucket name is
        // request input while the endpoint it is fetched
        // from is operator configuration, so the host
        // allowlist cannot speak for it. An entry naming
        // the bucket (or `*`) is required whether or not
        // credentials are configured; an empty list refuses
        // every bucket, the same fail-closed stance boot
        // takes for the credentialed combination.
        if !self
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
        self.read_body(resp, admission, &mode).await
    }

    /// Fetch a local file under the configured root. The
    /// candidate path is canonicalized before the containment
    /// check, so `..` segments and symlink escapes are refused.
    async fn load_file(&self, url: &str, mode: BodyMode<'_>) -> Result<LoadBody, LoadError> {
        use tokio::io::AsyncReadExt;
        let Some(root) = &self.sources.file_root else {
            return Err(LoadError::Forbidden(
                "file sources are not configured".to_string(),
            ));
        };
        let parsed =
            Url::parse(url).map_err(|e| LoadError::InvalidSource(format!("invalid url: {e}")))?;
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
            return Err(LoadError::InvalidSource(
                "invalid file source path".to_string(),
            ));
        }
        let candidate = root.join(decoded.trim_start_matches('/'));
        let canonical = tokio::fs::canonicalize(&candidate)
            .await
            .map_err(map_file_error)?;
        if !canonical.starts_with(root) {
            // Outside the root is the same answer as missing
            // inside it (issue #325): distinguishing the two
            // would make the response an oracle for what
            // exists on disk.
            return Err(LoadError::NotFound);
        }
        let mut file = tokio::fs::File::open(&canonical)
            .await
            .map_err(map_file_error)?;
        if matches!(mode, BodyMode::Probe(_)) {
            // A probe of a file needs only its head; the
            // total is the file's own size. The size cap
            // keeps the full load's too-large answer.
            let total = file.metadata().await.map_err(map_file_error)?.len();
            if total > self.max_bytes {
                return Err(LoadError::TooLarge {
                    limit: self.max_bytes,
                });
            }
            let mut head = Vec::new();
            let mut buf = [0u8; 8192];
            while head.len() < PROBE_HEAD_BYTES {
                let n = file.read(&mut buf).await.map_err(map_file_error)?;
                if n == 0 {
                    break;
                }
                let room = PROBE_HEAD_BYTES - head.len();
                head.extend_from_slice(&buf[..n.min(room)]);
            }
            return Ok(LoadBody::Probe(ProbeLoad {
                head,
                total_bytes: Some(total),
            }));
        }
        // A charged file read (issue #403) reserves the
        // file's size before reading; a local read does not
        // stall the way an origin can, and a file that grows
        // under the read charges the rest as it arrives.
        let mut part = match mode {
            BodyMode::Full(Some(charged)) => {
                let total = file.metadata().await.map_err(map_file_error)?.len();
                if total > self.max_bytes {
                    return Err(LoadError::TooLarge {
                        limit: self.max_bytes,
                    });
                }
                let total = usize::try_from(total).unwrap_or(usize::MAX);
                Some(BodyCharge::file(charged.charge, total).await?)
            }
            _ => None,
        };
        let mut body = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf).await.map_err(map_file_error)?;
            if n == 0 {
                break;
            }
            if (body.len() + n) as u64 > self.max_bytes {
                return Err(LoadError::TooLarge {
                    limit: self.max_bytes,
                });
            }
            if let Some(part) = &mut part {
                part.cover(body.len() + n).await?;
            }
            body.extend_from_slice(&buf[..n]);
        }
        if let Some(part) = part {
            part.finish(body.len());
        }
        Ok(LoadBody::Full(body))
    }

    /// Build a client whose DNS for the pinned host is forced
    /// to the address the policy validated (audit L-3). TLS
    /// still authenticates the original hostname. Proxies are
    /// disabled: a proxied request resolves at the proxy, so
    /// pinning through one would be meaningless. Clients are
    /// cached per (host, address) pair and reused across
    /// hops and fetches (issue #369): the pair is the whole
    /// key, so a client is only ever reused for the exact
    /// pin it was built for.
    fn pinned_client(&self, pin: &PinnedResolution) -> Result<Client, LoadError> {
        let key = (pin.host.clone(), pin.addr);
        if let Some(client) = self.pinned_clients.lock().unwrap().get(&key) {
            return Ok(client.clone());
        }
        let client = Client::builder()
            .timeout(self.timeout)
            .redirect(redirect::Policy::none())
            .no_proxy()
            .resolve(&pin.host, pin.addr)
            .build()
            .map_err(|e| LoadError::Network(e.to_string()))?;
        let mut cache = self.pinned_clients.lock().unwrap();
        if cache.len() >= MAX_PINNED_CLIENTS {
            cache.clear();
        }
        cache.insert(key, client.clone());
        Ok(client)
    }

    /// Run the source policy against `url`. No connection has been
    /// made yet when this runs. When the private-address check
    /// resolves a hostname, the validated address comes back so
    /// the caller can pin the connection to it.
    async fn check_policy(&self, url: &Url) -> Result<Option<PinnedResolution>, LoadError> {
        // The loader never presents credentials to an origin
        // (audit L-2): a signed URL must not make the server
        // authenticate anywhere with author-chosen userinfo.
        if !url.username().is_empty() || url.password().is_some() {
            return Err(LoadError::Forbidden(
                "source URL must not contain credentials".to_string(),
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| LoadError::Forbidden("source URL has no host".to_string()))?;
        let allowlisted = self
            .policy
            .allowed_suffixes
            .iter()
            .any(|s| suffix_matches(host, s));
        if !self.policy.allowed_suffixes.is_empty() && !allowlisted {
            return Err(LoadError::Forbidden(format!(
                "host {host} is not an allowed source"
            )));
        }
        // An allowlisted host is the operator's exception:
        // listing a host is how a deployment vouches for a
        // loopback or internal source, so the private-address
        // check (and the connection pinning that comes with
        // it) applies to every host the operator has not
        // listed.
        if self.policy.block_private && !allowlisted {
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
            // Hostnames (not IP literals) get pinned to the
            // first validated address for this connection.
            if bare.parse::<IpAddr>().is_err() {
                let port = url
                    .port()
                    .unwrap_or(if url.scheme() == "https" { 443 } else { 80 });
                if let Some(ip) = ips.first() {
                    return Ok(Some(PinnedResolution {
                        host: host.to_string(),
                        addr: std::net::SocketAddr::new(*ip, port),
                    }));
                }
            }
        }
        Ok(None)
    }
}

/// Whether `host` equals `suffix` or is a subdomain of it. The dot
/// boundary is what keeps `evil-example.com` from matching
/// `example.com`.
fn suffix_matches(host: &str, suffix: &str) -> bool {
    // Allocation-free form of `host == suffix ||
    // host.ends_with(".{suffix}")`: the part before the
    // suffix must end at a dot boundary.
    host == suffix
        || host
            .strip_suffix(suffix)
            .is_some_and(|rest| rest.ends_with('.'))
}

/// Non-public addresses are never legitimate public image
/// sources: loopback, private, link-local, unspecified, and the
/// remaining special-purpose ranges (shared/CGNAT, reserved,
/// multicast, broadcast, documentation, benchmarking, and the
/// protocol-assignment blocks). IPv6 forms that embed an IPv4
/// address (mapped, compatible, 6to4, NAT64, Teredo) are judged
/// by the IPv4 rules on the embedded address; the IPv6-native
/// special-purpose ranges (local-use NAT64, site-local,
/// benchmarking, discard-only) are blocked outright.
fn is_non_public(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_non_public_v4(ip),
        IpAddr::V6(ip) => is_non_public_v6(ip),
    }
}

fn is_non_public_v4(ip: &Ipv4Addr) -> bool {
    let octets = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || octets[0] == 0 // 0.0.0.0/8, "this network"
        || (octets[0] == 100 && (octets[1] & 0xC0) == 0x40) // 100.64.0.0/10, CGNAT
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0) // 192.0.0.0/24
        || ip.is_documentation()
        || (octets[0] == 198 && (octets[1] & 0xFE) == 18) // 198.18.0.0/15, benchmarking
        || (octets[0] & 0xF0) == 0xF0 // 240.0.0.0/4, reserved
        || ip.is_broadcast() // 255.255.255.255
        || ip.is_multicast() // 224.0.0.0/4
}

fn is_non_public_v6(ip: &Ipv6Addr) -> bool {
    // IPv4-mapped (::ffff:a.b.c.d) and IPv4-compatible
    // (::a.b.c.d) forms inherit the IPv4 classification.
    if let Some(ipv4) = ip.to_ipv4() {
        return is_non_public_v4(&ipv4);
    }
    let segments = ip.segments();
    // The remaining IPv4 embeddings (issue #364) are judged
    // by the IPv4 rules on the embedded address, the 6to4
    // pattern below: NAT64 (64:ff9b::/96) embeds its target
    // in the last two segments, and a Teredo address
    // (2001::/32) embeds its server in the third and fourth.
    if segments[0] == 0x0064 && segments[1] == 0xFF9B && segments[2] == 0 && segments[3] == 0 {
        let embedded = Ipv4Addr::new(
            (segments[6] >> 8) as u8,
            segments[6] as u8,
            (segments[7] >> 8) as u8,
            segments[7] as u8,
        );
        if is_non_public_v4(&embedded) {
            return true;
        }
    }
    if segments[0] == 0x2001 && segments[1] == 0 {
        let server = Ipv4Addr::new(
            (segments[2] >> 8) as u8,
            segments[2] as u8,
            (segments[3] >> 8) as u8,
            segments[3] as u8,
        );
        if is_non_public_v4(&server) {
            return true;
        }
    }
    // 6to4 (2002::/16) embeds the relay IPv4 address in the
    // second and third segments; a non-public relay makes the
    // address non-public.
    if segments[0] == 0x2002 {
        let relay = Ipv4Addr::new(
            (segments[1] >> 8) as u8,
            segments[1] as u8,
            (segments[2] >> 8) as u8,
            segments[2] as u8,
        );
        if is_non_public_v4(&relay) {
            return true;
        }
    }
    ip.is_loopback()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (segments[0] == 0x2001 && segments[1] == 0x0DB8) // 2001:db8::/32, documentation
        || (segments[0] == 0x0064 && segments[1] == 0xFF9B && segments[2] == 0x0001) // 64:ff9b:1::/48, local-use NAT64
        || (segments[0] & 0xFFC0) == 0xFEC0 // fec0::/10, site-local
        || (segments[0] == 0x2001 && segments[1] == 0x0002 && segments[2] == 0) // 2001:2::/48, benchmarking
        || (segments[0] == 0x0100 && segments[1] == 0 && segments[2] == 0 && segments[3] == 0) // 100::/64, discard-only
}

/// Whether any `/`-separated component of `path` is a dot
/// segment (exactly `.` or `..`). Dots inside a segment name
/// (`a.b`, `c..d`, `...`) are ordinary key content, not
/// segments.
fn has_dot_segment(path: &str) -> bool {
    path.split('/')
        .any(|segment| segment == "." || segment == "..")
}

/// Split the `bucket/key` remainder of a bucket URL. Both parts
/// are required; the key may itself contain slashes.
fn split_bucket_key<'a>(rest: &'a str, scheme: &str) -> Result<(&'a str, &'a str), LoadError> {
    let invalid = || {
        LoadError::InvalidSource(format!(
            "invalid {scheme} url: expected {scheme}://bucket/key"
        ))
    };
    let (bucket, key) = rest.split_once('/').ok_or_else(invalid)?;
    if bucket.is_empty() || key.is_empty() {
        return Err(invalid());
    }
    if !is_valid_bucket_name(bucket) {
        return Err(LoadError::Forbidden(format!(
            "invalid bucket name: {bucket}"
        )));
    }
    Ok((bucket, key))
}

/// Whether `bucket` is a name the S3 bucket naming rules
/// allow: 3 to 63 characters of lowercase letters, digits,
/// dots, and hyphens, starting and ending with a letter or
/// digit, with no adjacent dots and no dot beside a hyphen,
/// and not spelled as an IPv4 address. The loader
/// interpolates the bucket into the request host for AWS,
/// so a bucket carrying a `?` or another host-breaking
/// character would move the signed request, and the
/// credentials and session token on it, to a host the
/// operator never named (audit R2-20). Enforced at parse
/// time, whatever the allowlist says (an allowlist of `*`
/// included); `gs://` sources share the check because they
/// are served through the same S3-compatible contract.
fn is_valid_bucket_name(bucket: &str) -> bool {
    let bytes = bucket.as_bytes();
    if bytes.len() < 3 || bytes.len() > 63 {
        return false;
    }
    if !bytes
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-' || *b == b'.')
    {
        return false;
    }
    let edge_ok = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
    if !edge_ok(bytes[0]) || !edge_ok(bytes[bytes.len() - 1]) {
        return false;
    }
    if bucket.contains("..") || bucket.contains(".-") || bucket.contains("-.") {
        return false;
    }
    // Not an IPv4 address spelling: four numeric labels.
    let labels: Vec<&str> = bucket.split('.').collect();
    if labels.len() == 4
        && labels.iter().all(|label| {
            !label.is_empty()
                && label.bytes().all(|b| b.is_ascii_digit())
                && label.parse::<u32>().is_ok_and(|n| n <= 255)
        })
    {
        return false;
    }
    true
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

/// Map a file-source io failure: a missing file is the
/// uniform not-found answer (issue #325); anything else
/// keeps the read-failure mapping it has today.
fn map_file_error(e: std::io::Error) -> LoadError {
    if e.kind() == std::io::ErrorKind::NotFound {
        LoadError::NotFound
    } else {
        LoadError::Network(format!("cannot read file source: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
                    // Consume the request before answering.
                    // Answering on accept races the client:
                    // unread request bytes left in the receive
                    // buffer can reset the connection on close
                    // and make the client discard the response
                    // (issue #257).
                    let mut buf = [0u8; 1024];
                    let _ = socket.read(&mut buf).await;
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
    async fn userinfo_sources_are_refused() {
        // Audit L-2: refused by policy before any connection,
        // so the unroutable port never matters.
        let loader = Loader::with_defaults().unwrap();
        let err = loader
            .load("http://user:pass@127.0.0.1:1/img.jpg")
            .await
            .unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
    }

    #[tokio::test]
    async fn pinned_client_forces_the_validated_address() {
        // Audit L-3: a client pinned for an unresolvable name
        // connects to the validated address anyway (and would
        // authenticate TLS for the original name). This test's
        // server reads the request before answering, like the
        // shared fixture (answering on accept races a client
        // whose request lands after the response; issue #257).
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let _ = socket.write_all(&ok_image_response()).await;
        });
        let loader = Loader::with_defaults().unwrap();
        let pin = PinnedResolution {
            host: "pinned.invalid".to_string(),
            addr: std::net::SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), port),
        };
        let client = loader.pinned_client(&pin).unwrap();
        let resp = client
            .get(format!("http://pinned.invalid:{port}/img.jpg"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[test]
    fn pinned_clients_are_cached_per_host_and_address() {
        // Issue #369: the same pin reuses one client; a
        // different address for the same host (a fresh
        // validation) is a different client, never the
        // cached one.
        let loader = Loader::with_defaults().unwrap();
        let pin = |port: u16| PinnedResolution {
            host: "pinned.invalid".to_string(),
            addr: std::net::SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), port),
        };
        loader.pinned_client(&pin(9101)).unwrap();
        loader.pinned_client(&pin(9101)).unwrap();
        assert_eq!(loader.pinned_clients.lock().unwrap().len(), 1);
        loader.pinned_client(&pin(9102)).unwrap();
        assert_eq!(loader.pinned_clients.lock().unwrap().len(), 2);
    }

    #[test]
    fn errors_render() {
        assert_eq!(LoadError::NotFound.to_string(), "source not found");
        assert_eq!(
            LoadError::BadStatus(404).to_string(),
            "source answered HTTP 404"
        );
        assert_eq!(
            LoadError::TooLarge { limit: 10 }.to_string(),
            "source exceeded the 10-byte limit"
        );
        assert_eq!(
            LoadError::InvalidSource("unsupported scheme: ftp".to_string()).to_string(),
            "invalid source: unsupported scheme: ftp"
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
    async fn video_content_type_needs_the_media_opt_in() {
        // Issue #214: a `video/*` Content-Type is refused
        // by the plain load and accepted by the media load
        // a frame request uses; the opt-in widens nothing
        // else.
        let body = b"video";
        let video_response = || {
            response(
                "200 OK",
                "Content-Type: video/mp4\r\nContent-Length: 5\r\n",
                body,
            )
        };
        let loader = Loader::with_defaults().unwrap();
        let base = spawn_server(Respond::Bytes(video_response())).await;
        assert_eq!(
            fetch(&loader, &base).await,
            Err(LoadError::WrongContentType(Some("video/mp4".to_string())))
        );
        let base = spawn_server(Respond::Bytes(video_response())).await;
        let video = MediaAdmission {
            video: true,
            pdf: false,
        };
        assert_eq!(loader.load_media(&base, video).await.unwrap(), b"video");
        let resp = response(
            "200 OK",
            "Content-Type: text/html\r\nContent-Length: 5\r\n",
            body,
        );
        let base = spawn_server(Respond::Bytes(resp)).await;
        assert_eq!(
            loader.load_media(&base, video).await,
            Err(LoadError::WrongContentType(Some("text/html".to_string())))
        );
    }

    #[tokio::test]
    async fn pdf_content_type_needs_the_media_opt_in() {
        // Issue #215: an `application/pdf` Content-Type is
        // refused by the plain load and accepted by the
        // media load a page request uses; the video
        // admission does not cover it, and it widens
        // nothing else.
        let body = b"%PDF-";
        let pdf_response = || {
            response(
                "200 OK",
                "Content-Type: application/pdf\r\nContent-Length: 5\r\n",
                body,
            )
        };
        let loader = Loader::with_defaults().unwrap();
        let base = spawn_server(Respond::Bytes(pdf_response())).await;
        assert_eq!(
            fetch(&loader, &base).await,
            Err(LoadError::WrongContentType(Some(
                "application/pdf".to_string()
            )))
        );
        let pdf = MediaAdmission {
            video: false,
            pdf: true,
        };
        let base = spawn_server(Respond::Bytes(pdf_response())).await;
        assert_eq!(loader.load_media(&base, pdf).await.unwrap(), b"%PDF-");
        // The video admission alone does not admit a PDF.
        let video_only = MediaAdmission {
            video: true,
            pdf: false,
        };
        let base = spawn_server(Respond::Bytes(pdf_response())).await;
        assert_eq!(
            loader.load_media(&base, video_only).await,
            Err(LoadError::WrongContentType(Some(
                "application/pdf".to_string()
            )))
        );
        // A lookalike type is not the PDF type.
        let resp = response(
            "200 OK",
            "Content-Type: application/pdfx\r\nContent-Length: 5\r\n",
            body,
        );
        let base = spawn_server(Respond::Bytes(resp)).await;
        assert_eq!(
            loader.load_media(&base, pdf).await,
            Err(LoadError::WrongContentType(Some(
                "application/pdfx".to_string()
            )))
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
    async fn origin_404_is_not_found() {
        // Issue #325: an origin answering 404 means the
        // source is not there, not that the gateway failed.
        let resp = response("404 Not Found", "Content-Length: 0\r\n", b"");
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(fetch(&loader, &base).await, Err(LoadError::NotFound));
    }

    #[tokio::test]
    async fn rejects_bad_status() {
        let resp = response("500 Internal Server Error", "Content-Length: 0\r\n", b"");
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::with_defaults().unwrap();
        assert_eq!(fetch(&loader, &base).await, Err(LoadError::BadStatus(500)));
    }

    #[tokio::test]
    async fn refused_connection_is_a_network_error() {
        // Issue #325: the transport half of the split. A
        // port nothing listens on refuses the connection,
        // which stays a gateway-class failure, never a 404.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let loader = Loader::with_defaults().unwrap();
        let err = loader
            .load(&format!("http://127.0.0.1:{port}/img.jpg"))
            .await
            .unwrap_err();
        assert!(matches!(err, LoadError::Network(_)), "{err:?}");
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

    /// Spawn a server that answers every request after a
    /// fixed delay with a redirect to the next hop, ending
    /// in an image after `hops` redirects.
    async fn spawn_redirect_chain(hops: u32, delay: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = socket.read(&mut buf).await;
                    let request = String::from_utf8_lossy(&buf);
                    let n: u32 = request
                        .split("?n=")
                        .nth(1)
                        .and_then(|rest| {
                            rest.split(|c: char| !c.is_ascii_digit())
                                .next()
                                .and_then(|digits| digits.parse().ok())
                        })
                        .unwrap_or(0);
                    sleep(delay).await;
                    let resp = if n < hops {
                        response(
                            "302 Found",
                            &format!("Location: /hop?n={}\r\nContent-Length: 0\r\n", n + 1),
                            b"",
                        )
                    } else {
                        ok_image_response()
                    };
                    let _ = socket.write_all(&resp).await;
                });
            }
        });
        format!("http://{addr}")
    }

    /// Spawn a server that answers with the given headers
    /// and head bytes, then holds the connection open: the
    /// rest of the declared body never arrives.
    async fn spawn_drip_head_server(headers: &str, head: &[u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let head = head.to_vec();
        let headers = headers.to_string();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let _ = socket
                .write_all(
                    format!("HTTP/1.1 200 OK\r\n{headers}Connection: close\r\n\r\n").as_bytes(),
                )
                .await;
            let _ = socket.write_all(&head).await;
            sleep(Duration::from_secs(3600)).await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn probe_load_stops_once_the_head_settles() {
        // The origin declares a 1 MiB body, sends 64 bytes,
        // then drips forever. A full load would hang to the
        // loader timeout; the probe returns as soon as its
        // predicate settles on the head, with the declared
        // total.
        let mut head = b"\x89PNG\r\n\x1a\n".to_vec();
        head.resize(64, 0);
        let base = spawn_drip_head_server(
            "Content-Type: image/png\r\nContent-Length: 1048576\r\n",
            &head,
        )
        .await;
        let loader = Loader::new(10 * 1024 * 1024, Duration::from_secs(30)).unwrap();
        let started = std::time::Instant::now();
        let probe = loader
            .load_probe(
                &format!("{base}/img.png"),
                MediaAdmission::default(),
                &|head: &[u8]| head.len() >= 33 && head.starts_with(b"\x89PNG"),
            )
            .await
            .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert!(probe.head.len() >= 33);
        assert_eq!(probe.total_bytes, Some(1_048_576));
    }

    #[tokio::test]
    async fn probe_load_counts_a_body_that_ends_inside_the_head() {
        // No declared length: a body that ends inside the
        // head states its total by having ended.
        let body = b"\xff\xd8\xff\xd9";
        let resp = response("200 OK", "Content-Type: image/jpeg\r\n", body);
        let base = spawn_server(Respond::Bytes(resp)).await;
        let loader = Loader::with_defaults().unwrap();
        let probe = loader
            .load_probe(
                &format!("{base}/img.jpg"),
                MediaAdmission::default(),
                &|_head: &[u8]| false,
            )
            .await
            .unwrap();
        assert_eq!(probe.head, body);
        assert_eq!(probe.total_bytes, Some(4));
    }

    #[tokio::test]
    async fn probe_load_keeps_the_too_large_answer() {
        // A declared length past the cap is the full load's
        // too-large refusal, answered from the headers
        // without waiting for the body.
        let base = spawn_drip_head_server(
            "Content-Type: image/png\r\nContent-Length: 200\r\n",
            b"\x89PNG\r\n\x1a\n",
        )
        .await;
        let loader = Loader::new(100, Duration::from_secs(30)).unwrap();
        let started = std::time::Instant::now();
        let result = loader
            .load_probe(
                &format!("{base}/img.png"),
                MediaAdmission::default(),
                &|_head: &[u8]| false,
            )
            .await;
        assert_eq!(result, Err(LoadError::TooLarge { limit: 100 }));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn the_timeout_is_a_total_across_redirect_hops() {
        // Five hops at 300 ms each: every individual request
        // answers well inside a 1-second per-request budget,
        // so a per-hop timeout would let the chain run to
        // ~1.8 s and succeed. The loader timeout is a total:
        // the load fails as a timeout at ~1 s instead.
        let base = spawn_redirect_chain(5, Duration::from_millis(300)).await;
        let loader = Loader::new(1024, Duration::from_secs(1)).unwrap();
        let started = std::time::Instant::now();
        let result = loader.load(&format!("{base}/hop?n=0")).await;
        assert_eq!(result, Err(LoadError::Timeout));
        assert!(
            started.elapsed() < Duration::from_millis(1500),
            "chain ran for {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn source_text_the_loader_cannot_fetch_is_an_invalid_source() {
        // Issue #467: the caller's own URL is at fault, not
        // an origin, so these are InvalidSource (a 400), not
        // Network (a 502). Hostile spellings included: none
        // reaches the network, and none panics. The ftp
        // source names a port that is listening, and the
        // listener sees no connection.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let long = format!("http://{}", "\u{0}".repeat(64 * 1024));
        let sources = [
            format!("ftp://127.0.0.1:{port}/a.jpg"),
            "ftp://8.8.8.8/a.jpg".to_string(),
            "notaurl".to_string(),
            String::new(),
            "http://".to_string(),
            "https://[::1/a.jpg".to_string(),
            "http://example.com:99999/a.jpg".to_string(),
            "http://exa mple.com/a.jpg".to_string(),
            "javascript:alert(1)".to_string(),
            "data:image/png;base64,iVBORw0KGgo=".to_string(),
            "S3://bucket/key.jpg".to_string(),
            "s3://".to_string(),
            "s3://bucketonly".to_string(),
            "s3://bucket/".to_string(),
            "gs://x".to_string(),
            "gs:///key".to_string(),
            long,
        ];
        let loader = Loader::with_defaults().unwrap();
        for source in &sources {
            let err = loader.load(source).await.unwrap_err();
            let shown: String = source.chars().take(40).collect();
            assert!(
                matches!(err, LoadError::InvalidSource(_)),
                "{shown}: {err:?}"
            );
        }
        assert!(
            listener.accept().is_err(),
            "an unsupported scheme never connects"
        );
    }

    #[tokio::test]
    async fn a_redirect_to_another_scheme_stays_a_network_error() {
        // Issue #467: only the first hop is the caller's
        // text. A redirect to a scheme the loader does not
        // fetch is the origin's answer, a gateway failure.
        let redirect_resp = response(
            "302 Found",
            "Location: ftp://127.0.0.1/a.jpg\r\nContent-Length: 0\r\n",
            b"",
        );
        let base = spawn_server(Respond::Bytes(redirect_resp)).await;
        let loader = Loader::with_defaults().unwrap();
        let err = fetch(&loader, &base).await.unwrap_err();
        assert!(matches!(err, LoadError::Network(_)), "{err:?}");
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
        // Plain IPv4: every special-purpose range is blocked.
        for s in [
            "127.0.0.1",       // loopback
            "127.255.0.1",     // loopback, top of 127.0.0.0/8
            "10.1.2.3",        // private
            "172.16.0.5",      // private
            "172.31.255.255",  // private, top of 172.16.0.0/12
            "192.168.1.1",     // private
            "169.254.1.1",     // link-local
            "0.0.0.0",         // unspecified
            "0.0.0.1",         // 0.0.0.0/8, "this network"
            "0.255.255.255",   // 0.0.0.0/8, top
            "100.64.0.1",      // CGNAT, bottom of 100.64.0.0/10
            "100.127.255.255", // CGNAT, top of 100.64.0.0/10
            "192.0.0.1",       // 192.0.0.0/24, protocol assignments
            "192.0.0.170",     // 192.0.0.0/24, NAT64/DNS64 well-known
            "192.0.2.1",       // documentation TEST-NET-1
            "198.51.100.1",    // documentation TEST-NET-2
            "203.0.113.1",     // documentation TEST-NET-3
            "198.18.0.1",      // benchmarking, bottom of 198.18.0.0/15
            "198.19.255.255",  // benchmarking, top of 198.18.0.0/15
            "240.0.0.1",       // reserved, bottom of 240.0.0.0/4
            "250.1.2.3",       // reserved
            "255.255.255.255", // broadcast
            "224.0.0.1",       // multicast, bottom of 224.0.0.0/4
            "239.255.255.255", // multicast, top
        ] {
            assert!(is_non_public(&s.parse().unwrap()), "{s}");
        }
        // IPv6 spellings: native ranges, plus the IPv4-embedding
        // forms (mapped, compatible, 6to4) of blocked IPv4 ranges.
        for s in [
            "::1",              // loopback
            "::",               // unspecified
            "fe80::1",          // link-local
            "fd00::1",          // unique local
            "fc00::1",          // unique local
            "ff02::1",          // multicast
            "2001:db8::1",      // documentation
            "::ffff:127.0.0.1", // mapped loopback
            "::ffff:10.0.0.1",  // mapped private
            "::ffff:192.168.1.1",
            "::ffff:169.254.1.1",
            "::ffff:0.0.0.0",
            "::ffff:0.0.0.1",    // mapped 0.0.0.0/8
            "::ffff:100.64.0.1", // mapped CGNAT
            "::ffff:192.0.0.1",
            "::ffff:192.0.2.1",
            "::ffff:198.18.0.1",
            "::ffff:240.0.0.1",
            "::ffff:224.0.0.1",
            "::ffff:255.255.255.255",
            "::127.0.0.1",     // compatible loopback
            "::10.0.0.1",      // compatible private
            "2002:7f00:1::",   // 6to4, relay 127.0.0.1
            "2002:a00:1::",    // 6to4, relay 10.0.0.1
            "2002:6440:1::",   // 6to4, relay 100.64.0.1
            "2002:c000:201::", // 6to4, relay 192.0.2.1
            // The remaining embeddings and IPv6-native
            // special-purpose ranges (issue #364).
            "64:ff9b::7f00:1",    // NAT64, embeds 127.0.0.1
            "64:ff9b::a9fe:101",  // NAT64, embeds 169.254.1.1
            "64:ff9b::a00:1",     // NAT64, embeds 10.0.0.1
            "64:ff9b:1::1",       // local-use NAT64, 64:ff9b:1::/48
            "64:ff9b:1::808:808", // local-use NAT64, public tail or not
            "2001:0:7f00:1::",    // Teredo, server 127.0.0.1
            "2001:0:a00:1::",     // Teredo, server 10.0.0.1
            "2001:0:c633:6401::", // Teredo, server 198.51.100.1 (documentation)
            "fec0::1",            // site-local, bottom of fec0::/10
            "feff:ffff::1",       // site-local, top of fec0::/10
            "2001:2::1",          // benchmarking, 2001:2::/48
            "2001:2:0:ffff::1",   // benchmarking, top of 2001:2::/48
            "100::1",             // discard-only, 100::/64
            "100::ffff:1",        // discard-only, top of 100::/64
        ] {
            assert!(is_non_public(&s.parse().unwrap()), "{s}");
        }
        // Public addresses stay allowed, including the public
        // spellings of the IPv4-embedding forms and addresses just
        // outside the blocked ranges.
        for s in [
            "8.8.8.8",
            "1.1.1.1",
            "11.0.0.1",
            "172.32.0.1",      // just above 172.16.0.0/12
            "100.63.255.255",  // just below 100.64.0.0/10
            "100.128.0.1",     // just above 100.64.0.0/10
            "192.0.1.1",       // just above 192.0.0.0/24
            "198.17.255.255",  // just below 198.18.0.0/15
            "198.20.0.1",      // just above 198.18.0.0/15
            "223.255.255.255", // just below 224.0.0.0/4
            "2606:4700:4700::1111",
            "2001:4860:4860::8888",
            "::ffff:8.8.8.8", // mapped public
            "::8.8.8.8",      // compatible public
            "2002:808:808::", // 6to4, relay 8.8.8.8
            // Embedding controls (issue #364): a public
            // embedded address is judged public, exactly
            // like the forms above, and addresses just
            // outside the blocked IPv6-native ranges stay
            // allowed.
            "64:ff9b::808:808", // NAT64, embeds 8.8.8.8
            "2001:0:808:808::", // Teredo, server 8.8.8.8
            "64:ff9b:2::1",     // just outside both NAT64 ranges
            "2001:3::1",        // just above 2001:2::/48
            "100:0:0:1::1",     // just above 100::/64
            "fbff::1",          // just below fec0::/10
        ] {
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
    async fn allowlisted_hosts_are_exempt_from_private_blocking() {
        // The allowlist is the operator's exception path: a
        // listed host may resolve to a non-public address
        // even with private blocking on (loopback and
        // internal sources are the point of listing them).
        let base = spawn_server(Respond::Bytes(ok_image_response())).await;
        let listed = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec!["127.0.0.1".to_string()], true));
        assert!(fetch(&listed, &base).await.is_ok());

        // The exemption is the listing, not the address
        // class: the same loopback server under a policy
        // that lists some other host is still refused.
        let base = spawn_server(Respond::Bytes(ok_image_response())).await;
        let unlisted = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec!["example.com".to_string()], true));
        let err = fetch(&unlisted, &base).await.unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
    }

    #[tokio::test]
    async fn private_blocking_admits_public_literals() {
        // The gate itself is what is pinned here: a public
        // literal passes the policy check under blocking
        // (the load itself would need a reachable host).
        let blocking = Loader::with_defaults()
            .unwrap()
            .with_policy(SourcePolicy::new(vec![], true));
        for url in ["http://8.8.8.8/img.jpg", "http://1.1.1.1/img.jpg"] {
            let parsed = Url::parse(url).unwrap();
            assert!(blocking.check_policy(&parsed).await.is_ok(), "{url}");
        }
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
            allowed_buckets: vec!["photos".to_string()],
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
            allowed_buckets: vec!["photos".to_string()],
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
            allowed_buckets: vec!["media".to_string()],
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
    async fn bucket_sources_need_an_allowlist_entry() {
        // No credentials and no allowlist: every bucket is
        // refused, for both schemes, before any request is
        // made. The bucket allowlist is the only gate that
        // can speak for a bucket source, since the endpoint
        // is operator configuration, not request input.
        let loader = Loader::with_defaults().unwrap();
        for url in ["s3://photos/pic.jpg", "gs://photos/pic.jpg"] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
        }
        // The `*` entry is the deliberate any-bucket opt-in:
        // with it, the same loads go through. (The capturing
        // fixture serves one connection, so each scheme gets
        // its own.)
        for scheme in ["s3", "gs"] {
            let (base, _captured) = spawn_capturing_server().await;
            let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
                s3_endpoint: Some(base.clone()),
                gcs_endpoint: base,
                allowed_buckets: vec!["*".to_string()],
                ..Default::default()
            });
            let body = loader
                .load(&format!("{scheme}://photos/pic.jpg"))
                .await
                .unwrap();
            assert_eq!(body, b"\xff\xd8\xff\xd9", "{scheme}");
        }
    }

    #[tokio::test]
    async fn dot_segment_keys_are_refused_for_both_schemes() {
        // Credentials and an allowlist naming the first bucket
        // are configured, so every URL below passes the allowlist
        // as written; the refusal must come from the dot-segment
        // check, before any request is signed (audit R2-05).
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            allowed_buckets: vec!["photos".to_string()],
            aws_creds: Some(test_creds()),
            gcs_creds: Some(test_creds()),
            ..Default::default()
        });
        let keys = [
            "../evil.jpg",
            "../../evil-bucket/secret.jpg",
            "a/../../evil.jpg",
            "a/b/..",
            "a/..",
            "..",
            ".",
            "./pic.jpg",
            "a/./pic.jpg",
            "a/.",
            "a//../../evil.jpg",
        ];
        for key in keys {
            for scheme in ["s3", "gs"] {
                let url = format!("{scheme}://photos/{key}");
                let err = loader.load(&url).await.unwrap_err();
                assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
            }
        }
        // A dot segment as the bucket is refused the same way.
        for url in ["s3://../pic.jpg", "gs://./pic.jpg"] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
        }
    }

    #[tokio::test]
    async fn dot_segment_key_sends_no_request() {
        // The canonical exploit spelling: as bucket `photos` it
        // passes the allowlist, and URL parsing would collapse
        // the key to a request for the other bucket. The load
        // must fail without the endpoint seeing any request.
        let (base, captured) = spawn_capturing_server().await;
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            gcs_endpoint: base,
            gcs_creds: Some(test_creds()),
            allowed_buckets: vec!["photos".to_string()],
            ..Default::default()
        });
        let err = loader
            .load("gs://photos/../../evil-bucket/secret.jpg")
            .await
            .unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn dotted_segment_names_still_load() {
        // Dots inside a segment name are ordinary key content:
        // none of these keys has a `.` or `..` segment, so each
        // loads and reaches the endpoint spelled as written.
        let keys = [
            "a.b/c..d/pic.jpg",
            "dir/.../pic.jpg",
            "dir/.hidden/pic.jpg",
            "dir/..hidden/pic.jpg",
            "dir/file..jpg",
        ];
        for key in keys {
            let (base, captured) = spawn_capturing_server().await;
            let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
                s3_endpoint: Some(base),
                allowed_buckets: vec!["photos".to_string()],
                ..Default::default()
            });
            let body = loader.load(&format!("s3://photos/{key}")).await.unwrap();
            assert_eq!(body, b"\xff\xd8\xff\xd9");
            let head = captured.lock().unwrap().clone();
            let first = head.lines().next().unwrap().to_string();
            assert!(first.starts_with(&format!("GET /photos/{key} ")), "{first}");

            let (base, captured) = spawn_capturing_server().await;
            let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
                gcs_endpoint: base,
                allowed_buckets: vec!["photos".to_string()],
                ..Default::default()
            });
            let body = loader.load(&format!("gs://photos/{key}")).await.unwrap();
            assert_eq!(body, b"\xff\xd8\xff\xd9");
            let head = captured.lock().unwrap().clone();
            let first = head.lines().next().unwrap().to_string();
            assert!(first.starts_with(&format!("GET /photos/{key} ")), "{first}");
        }
    }

    #[tokio::test]
    async fn malformed_bucket_urls_error() {
        let loader = Loader::with_defaults().unwrap();
        for url in ["s3://", "s3://bucketonly", "s3://bucket/", "gs:///key"] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::InvalidSource(_)), "{url}: {err:?}");
        }
    }

    /// A scratch tree: `root/` holds the servable files, a
    /// secret sits next to it outside the root.
    fn make_file_tree() -> (crate::test_support::ScratchDir, PathBuf) {
        let base = crate::test_support::scratch_dir("loader-test");
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
        // An escape answers exactly what a missing file
        // answers (issue #325), so the response cannot be
        // used to tell the two apart.
        for url in [
            "file:///link.jpg",
            "file:///linkdir/secret.jpg",
            "file:///%2e%2e/secret.jpg",
        ] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::NotFound), "{url}: {err:?}");
        }
        // Dot-segment escapes are collapsed by URL parsing into
        // the root and fail there as missing files; either way
        // the secret is never served.
        for url in ["file:///../secret.jpg", "file:///sub/../../secret.jpg"] {
            assert_eq!(loader.load(url).await, Err(LoadError::NotFound), "{url}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn file_sources_missing_unconfigured_and_oversized() {
        let (base, root) = make_file_tree();
        let loader = file_loader(&root, 1024);
        assert_eq!(
            loader.load("file:///nope.jpg").await,
            Err(LoadError::NotFound)
        );
        let small = file_loader(&root, 8);
        assert_eq!(
            small.load("file:///big.bin").await,
            Err(LoadError::TooLarge { limit: 8 })
        );
        let unconfigured = Loader::with_defaults().unwrap();
        let err = unconfigured.load("file:///pic.jpg").await.unwrap_err();
        assert!(matches!(err, LoadError::Forbidden(_)), "{err:?}");
        // A NUL in the path is the caller's malformed
        // locator (issue #467), never a read attempt.
        let err = loader.load("file:///pic%00.jpg").await.unwrap_err();
        assert!(matches!(err, LoadError::InvalidSource(_)), "{err:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn bucket_keys_and_percent_decoding_parse() {
        assert_eq!(
            split_bucket_key("bkt/dir/k.jpg", "s3").unwrap(),
            ("bkt", "dir/k.jpg")
        );
        assert!(split_bucket_key("bkt/", "s3").is_err());
        assert_eq!(percent_decode("/a%20b/%2e%2e/c+d"), "/a b/../c+d");
        assert_eq!(percent_decode("/plain%zz%"), "/plain%zz%");
    }

    #[test]
    fn bucket_names_follow_the_s3_rules() {
        // Valid: ordinary names, dotted names, hyphenated
        // names, the length bounds.
        for name in [
            "photos",
            "my.bucket-1",
            "a1b",
            "a.b",
            "0abc",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ] {
            assert!(is_valid_bucket_name(name), "{name}");
        }
        // Invalid: host-breaking characters above all (audit
        // R2-20), then the rest of the naming rules.
        for name in [
            "attacker.example?x",
            "bucket#frag",
            "bucket/key",
            "bucket name",
            "bucket_name",
            "UPPER",
            "ab",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "-bucket",
            "bucket-",
            ".bucket",
            "bucket.",
            "a..b",
            "a.-b",
            "a-.b",
            "192.168.0.1",
            "1.2.3.4",
        ] {
            assert!(!is_valid_bucket_name(name), "{name}");
        }
        assert!(split_bucket_key("attacker.example?x/pic.jpg", "s3").is_err());
    }

    #[tokio::test]
    async fn crafted_bucket_names_send_no_request() {
        // Audit R2-20: with the allowlist wide open (`*`) and
        // credentials carrying a session token, a bucket
        // string that would move the signed request off AWS
        // is refused at parse time; the endpoint sees no
        // request at all.
        let (base, captured) = spawn_capturing_server().await;
        let loader = Loader::with_defaults().unwrap().with_sources(SourceConfig {
            s3_endpoint: Some(base),
            allowed_buckets: vec!["*".to_string()],
            aws_creds: Some(test_creds()),
            ..Default::default()
        });
        for url in [
            "s3://attacker.example?x/pic.jpg",
            "s3://UPPER/pic.jpg",
            "s3://a..b/pic.jpg",
        ] {
            let err = loader.load(url).await.unwrap_err();
            assert!(matches!(err, LoadError::Forbidden(_)), "{url}: {err:?}");
        }
        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_source_charge_grows_shrinks_splits_and_releases() {
        let budget = SourceBudget::new(100);
        let wait = Duration::from_millis(50);
        let charge = budget.charge(wait);
        charge.grow_to(60).await.unwrap();
        assert_eq!((charge.bytes(), budget.in_use()), (60, 60));
        // Growing to what is held is free; shrinking never
        // grows.
        charge.grow_to(40).await.unwrap();
        charge.shrink_to(80);
        assert_eq!(charge.bytes(), 60);
        // More than is free: the wait runs out.
        let other = budget.charge(wait);
        assert_eq!(other.grow_to(41).await, Err(LoadError::Busy));
        // More than the whole budget can never be granted:
        // refused at once instead of waiting it out.
        let start = std::time::Instant::now();
        let big = budget.charge(Duration::from_secs(5));
        assert_eq!(big.grow_to(101).await, Err(LoadError::Busy));
        assert!(start.elapsed() < Duration::from_secs(1));
        // Splitting moves bytes into a part released on its
        // own.
        let part = charge.split_off(25);
        assert_eq!((charge.bytes(), part.bytes()), (35, 25));
        drop(part);
        assert_eq!(budget.in_use(), 35);
        charge.shrink_to(10);
        assert_eq!(budget.in_use(), 10);
        drop(charge);
        assert_eq!(budget.in_use(), 0);
        // A fresh loader's budget never sheds; the server
        // installs the configured one.
        assert_eq!(
            Loader::with_defaults().unwrap().source_budget().capacity(),
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }

    #[tokio::test]
    async fn a_request_waits_for_budget_once_in_total_not_per_load() {
        // The wait bound is the request's, across all its
        // loads: two waits that each run out take the one
        // budget of wait between them, not one each.
        let budget = SourceBudget::new(10);
        let held = budget.charge(Duration::ZERO);
        held.grow_to(10).await.unwrap();
        let charge = budget.charge(Duration::from_millis(300));
        let start = std::time::Instant::now();
        assert_eq!(charge.grow_to(5).await, Err(LoadError::Busy));
        assert_eq!(charge.grow_to(5).await, Err(LoadError::Busy));
        let waited = start.elapsed();
        assert!(waited < Duration::from_millis(550), "{waited:?}");
        // A waiter is granted as soon as bytes come free.
        let waiter = budget.charge(Duration::from_secs(5));
        let release = async {
            sleep(Duration::from_millis(100)).await;
            drop(held);
        };
        let (granted, ()) = tokio::join!(waiter.grow_to(8), release);
        assert_eq!(granted, Ok(()));
        assert_eq!(waiter.bytes(), 8);
    }

    #[tokio::test]
    async fn a_charged_load_keeps_exactly_its_body_and_a_failed_one_keeps_nothing() {
        let body = vec![7u8; 100_000];
        let ok = response(
            "200 OK",
            &format!(
                "Content-Type: image/png\r\nContent-Length: {}\r\n",
                body.len()
            ),
            &body,
        );
        let budget = SourceBudget::new(1 << 20);
        let loader = Loader::with_defaults()
            .unwrap()
            .with_source_budget(budget.clone());
        let charge = budget.charge(Duration::from_secs(1));
        let base = spawn_server(Respond::Bytes(ok)).await;
        let got = loader
            .load_media_charged(&format!("{base}/a.png"), MediaAdmission::default(), &charge)
            .await
            .unwrap();
        assert_eq!(got.len(), body.len());
        assert_eq!(charge.bytes(), body.len());
        // A body past the cap fails and releases its part,
        // keeping what the request already held.
        let small = Loader::new(1000, Duration::from_secs(5))
            .unwrap()
            .with_source_budget(budget.clone());
        let undeclared = response("200 OK", "Content-Type: image/png\r\n", &body);
        let base = spawn_server(Respond::Bytes(undeclared)).await;
        let err = small
            .load_media_charged(&format!("{base}/b.png"), MediaAdmission::default(), &charge)
            .await
            .unwrap_err();
        assert_eq!(err, LoadError::TooLarge { limit: 1000 });
        assert_eq!(charge.bytes(), body.len());
        assert_eq!(budget.in_use(), body.len());
        // So does a load the loader's timeout cancels while
        // its body is held by a stalled origin.
        let quick = Loader::new(1 << 20, Duration::from_millis(300))
            .unwrap()
            .with_source_budget(budget.clone());
        let base = spawn_stalling_server(1 << 19, 1000).await;
        let err = quick
            .load_media_charged(&format!("{base}/c.png"), MediaAdmission::default(), &charge)
            .await
            .unwrap_err();
        assert_eq!(err, LoadError::Timeout);
        assert_eq!(budget.in_use(), body.len());
        drop(charge);
        assert_eq!(budget.in_use(), 0);
    }

    /// An origin that answers with a declared length of
    /// `declared` bytes, sends `sent` of them, then holds
    /// the connection open without sending the rest.
    async fn spawn_stalling_server(declared: usize, sent: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = socket.read(&mut buf).await;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {declared}\r\n\r\n"
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(&vec![0u8; sent]).await;
                    sleep(Duration::from_secs(30)).await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_stalled_origin_holds_only_what_it_delivered() {
        // An origin that declares the whole cap, sends 1000
        // bytes, and stalls: the load holds its read-ahead
        // (here its whole declared length) only for its first
        // window, then only what arrived,
        // so a second request's load proceeds instead of
        // waiting out the stall (the coordinator's design
        // note on issue #403: up-front reservation would let
        // a few stalled origins starve every other request).
        const CAP: usize = 1 << 20;
        let budget = SourceBudget::new(CAP);
        let loader = Loader::new(CAP as u64, Duration::from_secs(20))
            .unwrap()
            .with_source_budget(budget.clone());
        let stalled = spawn_stalling_server(CAP, 1000).await;
        let stalled_load = {
            let loader = loader.clone();
            let budget = budget.clone();
            tokio::spawn(async move {
                let charge = budget.charge(Duration::from_secs(5));
                let result = loader
                    .load_media_charged(
                        &format!("{stalled}/s.png"),
                        MediaAdmission::default(),
                        &charge,
                    )
                    .await;
                (result.map(|b| b.len()), charge.bytes())
            })
        };
        // While the grant lasts, the stall holds the cap.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while budget.in_use() < CAP {
            assert!(std::time::Instant::now() < deadline, "no grant taken");
            sleep(Duration::from_millis(10)).await;
        }
        // A healthy load arriving now waits at most for the
        // grant to end, far short of the stall.
        let body = vec![1u8; 4000];
        let ok = response(
            "200 OK",
            "Content-Type: image/png\r\nContent-Length: 4000\r\n",
            &body,
        );
        let healthy = spawn_server(Respond::Bytes(ok)).await;
        let charge = budget.charge(Duration::from_secs(5));
        let start = std::time::Instant::now();
        let got = loader
            .load_media_charged(
                &format!("{healthy}/h.png"),
                MediaAdmission::default(),
                &charge,
            )
            .await
            .unwrap();
        let waited = start.elapsed();
        assert_eq!(got.len(), 4000);
        assert!(
            waited < BODY_GRANT + Duration::from_millis(500),
            "{waited:?}"
        );
        // The stall now holds exactly what it delivered.
        assert_eq!(budget.in_use(), 1000 + 4000);
        stalled_load.abort();
    }

    /// The burst the source budget exists for (issue #403):
    /// `loads` concurrent distinct loads from an origin that
    /// takes 100 ms to answer, each holding its body for 50
    /// ms afterwards, as a render would. Returns the most
    /// body bytes held at once.
    async fn budgeted_burst(budget: SourceBudget, loads: usize) -> usize {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        const BODY: usize = 48 * 1024;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = socket.read(&mut buf).await;
                    sleep(Duration::from_millis(100)).await;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {BODY}\r\nConnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(&vec![0u8; BODY]).await;
                });
            }
        });
        let loader = Loader::new(64 * 1024, Duration::from_secs(10))
            .unwrap()
            .with_source_budget(budget.clone());
        let held = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for i in 0..loads {
            let loader = loader.clone();
            let budget = budget.clone();
            let (held, peak) = (held.clone(), peak.clone());
            tasks.push(tokio::spawn(async move {
                let charge = budget.charge(Duration::from_secs(5));
                let body = loader
                    .load_media_charged(
                        &format!("http://{addr}/s{i}.png"),
                        MediaAdmission::default(),
                        &charge,
                    )
                    .await
                    .expect("every load fits within the wait");
                let now = held.fetch_add(body.len(), Ordering::SeqCst) + body.len();
                peak.fetch_max(now, Ordering::SeqCst);
                sleep(Duration::from_millis(50)).await;
                held.fetch_sub(body.len(), Ordering::SeqCst);
                drop(body);
                drop(charge);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(budget.in_use(), 0, "every charge came back");
        peak.load(Ordering::SeqCst)
    }

    #[tokio::test]
    async fn a_burst_of_loads_buffers_what_the_budget_holds_not_what_arrives() {
        const BUDGET: usize = 3 * 48 * 1024;
        // Unbudgeted, the bodies the burst holds at once grow
        // with the request count: all twelve overlap.
        let peak = budgeted_burst(SourceBudget::unlimited(), 12).await;
        assert!(peak > BUDGET, "without a budget the burst overlaps: {peak}");
        // Under a budget of three bodies, never more than
        // three are held at once, however many are queued,
        // and every load is still served.
        let peak = budgeted_burst(SourceBudget::new(BUDGET), 12).await;
        assert!(
            peak <= BUDGET,
            "{peak} bytes held under a {BUDGET} byte budget"
        );
    }

    /// An origin that sends every body as `chunks` writes of
    /// `chunk` bytes, `every` apart.
    async fn spawn_paced_server(chunks: usize, chunk: usize, every: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = socket.read(&mut buf).await;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        chunks * chunk
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    for i in 0..chunks {
                        if i > 0 {
                            sleep(every).await;
                        }
                        if socket.write_all(&vec![0u8; chunk]).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_body_that_keeps_up_keeps_its_grant_under_a_burst() {
        // Four loads of bodies one and a half read-aheads
        // long that take 1.5 s to arrive, under a budget of
        // two bodies: each passes its read-ahead, takes the
        // rest of its length while the others wait, and keeps
        // it though the body never arrives in one window, so
        // two are served at a time and all four are served. A
        // load that gave its grant back while its body was
        // still arriving would have to charge the rest behind
        // the loads still waiting for a grant of their own,
        // and, with nothing left to free, be shed half-read.
        const CHUNK: usize = 256 * 1024;
        const CHUNKS: usize = 6;
        const BODY: usize = CHUNK * CHUNKS;
        let every = Duration::from_millis(300);
        assert!(every * (CHUNKS as u32 - 1) > BODY_GRANT);
        const { assert!(BODY > BODY_READ_AHEAD) };
        let budget = SourceBudget::new(2 * BODY);
        let loader = Loader::new(2 << 20, Duration::from_secs(20))
            .unwrap()
            .with_source_budget(budget.clone());
        let base = spawn_paced_server(CHUNKS, CHUNK, every).await;
        let loads: Vec<_> = (0..4)
            .map(|i| {
                let (loader, budget, base) = (loader.clone(), budget.clone(), base.clone());
                tokio::spawn(async move {
                    let charge = budget.charge(Duration::from_secs(5));
                    let body = loader
                        .load_media_charged(
                            &format!("{base}/s{i}.png"),
                            MediaAdmission::default(),
                            &charge,
                        )
                        .await?;
                    // Held a moment, as a render would.
                    sleep(Duration::from_millis(50)).await;
                    Ok::<usize, LoadError>(body.len())
                })
            })
            .collect();
        for load in loads {
            assert_eq!(load.await.unwrap(), Ok(BODY));
        }
        assert_eq!(budget.in_use(), 0, "every charge came back");
    }

    #[tokio::test]
    async fn a_burst_of_bodies_past_the_read_ahead_all_finish() {
        // Eight fast bodies one and a half read-aheads long,
        // under a budget of two. If every waiting body could
        // reserve a read-ahead, three would fill the budget,
        // none would have room to take the rest of its
        // length, and all would wait on each other until they
        // shed. The read-ahead pool keeps that room, so the
        // bodies finish a few at a time and every one is
        // served.
        const BODY: usize = BODY_READ_AHEAD + BODY_READ_AHEAD / 2;
        let budget = SourceBudget::new(2 * BODY);
        let loader = Loader::new(2 << 20, Duration::from_secs(20))
            .unwrap()
            .with_source_budget(budget.clone());
        let base = spawn_paced_server(1, BODY, Duration::ZERO).await;
        let loads: Vec<_> = (0..8)
            .map(|i| {
                let (loader, budget, base) = (loader.clone(), budget.clone(), base.clone());
                tokio::spawn(async move {
                    let charge = budget.charge(Duration::from_secs(5));
                    let body = loader
                        .load_media_charged(
                            &format!("{base}/b{i}.png"),
                            MediaAdmission::default(),
                            &charge,
                        )
                        .await?;
                    sleep(Duration::from_millis(50)).await;
                    Ok::<usize, LoadError>(body.len())
                })
            })
            .collect();
        for load in loads {
            assert_eq!(load.await.unwrap(), Ok(BODY));
        }
        assert_eq!(budget.in_use(), 0, "every charge came back");
    }

    /// An origin that answers every request with a declared
    /// length of `declared` bytes, then sends one byte every
    /// `every` for as long as the connection stays open.
    /// Returns its base URL, the requests it has answered,
    /// and the body bytes it has sent.
    async fn spawn_trickling_server(
        declared: usize,
        every: Duration,
    ) -> (
        String,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let answered = Arc::new(AtomicUsize::new(0));
        let sent = Arc::new(AtomicUsize::new(0));
        {
            let (answered, sent) = (answered.clone(), sent.clone());
            tokio::spawn(async move {
                loop {
                    let Ok((mut socket, _)) = listener.accept().await else {
                        break;
                    };
                    let (answered, sent) = (answered.clone(), sent.clone());
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1024];
                        let _ = socket.read(&mut buf).await;
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
                        );
                        if socket.write_all(head.as_bytes()).await.is_err() {
                            return;
                        }
                        answered.fetch_add(1, Ordering::SeqCst);
                        loop {
                            sent.fetch_add(1, Ordering::SeqCst);
                            if socket.write_all(&[0]).await.is_err() {
                                return;
                            }
                            sleep(every).await;
                        }
                    });
                }
            });
        }
        (format!("http://{addr}"), answered, sent)
    }

    #[tokio::test]
    async fn trickling_bodies_cannot_starve_a_fast_load() {
        // Twenty-four loads whose origin declares the whole
        // fetch cap and then sends a byte every 100 ms, under
        // a budget of sixteen caps. Granted their declared
        // length for as long as any byte arrived, they would
        // hold the whole budget until their timeout and shed
        // every other load. Each reserves only a read-ahead,
        // and its first window, far below the rate it would
        // need, ends that grant: a fast load is served at
        // once, and the trickles, still open, hold no more
        // than the bytes they sent.
        use std::sync::atomic::Ordering;
        const CAP: usize = 2 << 20;
        const TRICKLES: usize = 24;
        let budget = SourceBudget::new(16 * CAP);
        let loader = Loader::new(CAP as u64, Duration::from_secs(20))
            .unwrap()
            .with_source_budget(budget.clone());
        let (trickling, answered, sent) =
            spawn_trickling_server(CAP, Duration::from_millis(100)).await;
        let trickles: Vec<_> = (0..TRICKLES)
            .map(|i| {
                let (loader, budget, base) = (loader.clone(), budget.clone(), trickling.clone());
                tokio::spawn(async move {
                    let charge = budget.charge(Duration::from_secs(5));
                    let _ = loader
                        .load_media_charged(
                            &format!("{base}/t{i}.png"),
                            MediaAdmission::default(),
                            &charge,
                        )
                        .await;
                })
            })
            .collect();
        // Every trickle has been answered and holds at least
        // a read-ahead of budget between them.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while answered.load(Ordering::SeqCst) < TRICKLES
            || budget.in_use() < TRICKLES * BODY_READ_AHEAD
        {
            assert!(
                std::time::Instant::now() < deadline,
                "the trickles never took their grants: {} bytes held",
                budget.in_use()
            );
            sleep(Duration::from_millis(10)).await;
        }
        // A fast load is served at once, a small one and one
        // past the read-ahead alike.
        for len in [4000, BODY_READ_AHEAD + BODY_READ_AHEAD / 2] {
            let body = vec![1u8; len];
            let ok = response(
                "200 OK",
                &format!("Content-Type: image/png\r\nContent-Length: {len}\r\n"),
                &body,
            );
            let fast = spawn_server(Respond::Bytes(ok)).await;
            let charge = budget.charge(Duration::from_secs(2));
            let start = std::time::Instant::now();
            let got = loader
                .load_media_charged(&format!("{fast}/f.png"), MediaAdmission::default(), &charge)
                .await;
            let waited = start.elapsed();
            assert_eq!(got.map(|b| b.len()), Ok(len), "after {waited:?}");
            assert!(waited < Duration::from_secs(1), "{waited:?}");
        }
        // Their first window over, the trickles hold no more
        // than they delivered.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            let (held, delivered) = (budget.in_use(), sent.load(Ordering::SeqCst));
            if held <= delivered {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{held} bytes held for {delivered} delivered"
            );
            sleep(Duration::from_millis(20)).await;
        }
        for trickle in trickles {
            trickle.abort();
            let _ = trickle.await;
        }
        assert_eq!(budget.in_use(), 0, "every charge came back");
    }

    #[tokio::test]
    async fn a_body_too_slow_to_finish_in_time_holds_only_what_it_received() {
        // A body of one read-ahead arriving at 128 KiB/s,
        // twice the floor, under a 4 s timeout: at that pace
        // it cannot arrive before the timeout cancels it, so
        // its first window ends its grant and, though bytes
        // keep coming, it holds only what has arrived. A fast
        // load fits beside it in a budget of that one body.
        const CHUNK: usize = 32 * 1024;
        const BODY: usize = BODY_READ_AHEAD;
        let budget = SourceBudget::new(BODY);
        let loader = Loader::new(BODY as u64, Duration::from_secs(4))
            .unwrap()
            .with_source_budget(budget.clone());
        let slow = spawn_paced_server(BODY / CHUNK, CHUNK, Duration::from_millis(250)).await;
        let slow_load = {
            let (loader, budget) = (loader.clone(), budget.clone());
            tokio::spawn(async move {
                let charge = budget.charge(Duration::from_secs(5));
                loader
                    .load_media_charged(
                        &format!("{slow}/s.png"),
                        MediaAdmission::default(),
                        &charge,
                    )
                    .await
                    .map(|b| b.len())
            })
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while budget.in_use() < BODY {
            assert!(std::time::Instant::now() < deadline, "no grant taken");
            sleep(Duration::from_millis(10)).await;
        }
        let start = std::time::Instant::now();
        // By 2.5 s at most 320 KiB has arrived.
        while budget.in_use() > BODY / 2 {
            assert!(
                start.elapsed() < Duration::from_millis(2500),
                "{} bytes still held",
                budget.in_use()
            );
            sleep(Duration::from_millis(20)).await;
        }
        assert!(!slow_load.is_finished(), "the slow body is still arriving");
        let body = vec![1u8; 4000];
        let ok = response(
            "200 OK",
            "Content-Type: image/png\r\nContent-Length: 4000\r\n",
            &body,
        );
        let fast = spawn_server(Respond::Bytes(ok)).await;
        let charge = budget.charge(Duration::from_secs(1));
        let got = loader
            .load_media_charged(&format!("{fast}/f.png"), MediaAdmission::default(), &charge)
            .await;
        assert_eq!(got.map(|b| b.len()), Ok(4000));
        drop(charge);
        assert_eq!(slow_load.await.unwrap(), Err(LoadError::Timeout));
        assert_eq!(budget.in_use(), 0, "every charge came back");
    }
}
