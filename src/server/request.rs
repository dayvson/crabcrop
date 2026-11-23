//! The image route (issue #741): `handle` and `serve`, from
//! the request to its flight; the record each request owes
//! the log and the metrics; the request ID; the client hints
//! the route reads and the `Vary` it names; the cache key; and
//! the private marking of a hint-shaped answer.
//!
//! server.rs re-exports what the router and the tests use.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::Response;

use super::admission::{
    Credential, Negotiated, authenticate, negotiate, refuse_own_fetch, resolve_request,
};
use super::flight::run_flight;
use super::miss::MissWork;
use super::{AppState, MissProgress, has_directive};
use crate::observe::RequestRecord;
use crate::url::ImageFormat;

#[cfg(doc)]
use super::admission::Refusal;
#[cfg(doc)]
use super::flight::finish_started_miss;

/// The DPR hint headers [`dpr_hint`] reads, the
/// structured one first.
pub(in crate::server) const DPR_HINT_HEADERS: [&str; 2] = ["Sec-CH-DPR", "DPR"];

/// The width hint headers [`width_hint`] reads, in the
/// order it prefers them.
pub(in crate::server) const WIDTH_HINT_HEADERS: [&str; 4] = [
    "Sec-CH-Width",
    "Width",
    "Sec-CH-Viewport-Width",
    "Viewport-Width",
];

/// The header [`save_data_hint`] reads.
pub(in crate::server) const SAVE_DATA_HEADER: &str = "Save-Data";

/// The `Vary` of every image-route answer while the server
/// reads client hints: `Accept` and every header the hint
/// readers above read, since each can change the bytes.
pub(in crate::server) const VARY_WITH_HINTS: &str = "Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data";

/// The `Vary` of every image-route answer with client hints
/// off (`CRABCROP_CLIENT_HINTS=0`): no hint header is read,
/// so only `Accept` can change an answer.
pub(in crate::server) const VARY_WITHOUT_HINTS: &str = "Accept";

/// The `Vary` an image-route answer carries, success or
/// error: an error's body follows `Accept` (plain text or
/// the JSON envelope) and can follow a hint (a hinted size
/// that runs past a limit or the deadline), just as an
/// image does.
pub(in crate::server) fn image_vary(client_hints: bool) -> axum::http::HeaderValue {
    axum::http::HeaderValue::from_static(if client_hints {
        VARY_WITH_HINTS
    } else {
        VARY_WITHOUT_HINTS
    })
}

/// The `DPR` client hint, when it names one of the settled
/// factors. Anything else (absent, unparsable, out of set) is
/// no hint: hints are advisory, never an error. A factor
/// that would push the output past its limits steps down
/// in the engine rather than failing the request (issue
/// #529).
pub(in crate::server) fn dpr_hint(headers: &HeaderMap) -> Option<f32> {
    let [first, legacy] = DPR_HINT_HEADERS;
    let raw = headers
        .get(first)
        .or_else(|| headers.get(legacy))
        .and_then(|v| v.to_str().ok())?;
    raw.trim()
        .parse::<f32>()
        .ok()
        .filter(|f| crate::url::DPR_FACTORS.contains(f))
}

/// The width client hint: the display width the client
/// wants, from the most specific header present
/// (`Sec-CH-Width`, then the legacy `Width`, then the
/// viewport widths). Advisory like every hint: an absent,
/// unparsable, or out-of-range value is no hint, never an
/// error.
pub(in crate::server) fn width_hint(headers: &HeaderMap) -> Option<u32> {
    for name in WIDTH_HINT_HEADERS {
        let Some(raw) = headers.get(name).and_then(|v| v.to_str().ok()) else {
            continue;
        };
        if let Ok(w) = raw.trim().parse::<u32>()
            && (1..=crate::url::MAX_DIMENSION).contains(&w)
        {
            return Some(w);
        }
    }
    None
}

/// The widest width a client hint may resolve to (issue
/// #403). Hints are client-controlled and the hinted width
/// joins the cache key, so without a ceiling one URL fans
/// out into a key (and a render) per quantization step all
/// the way to the parser's dimension cap. 3840 is the UHD
/// display width; the DPR hint still multiplies on top,
/// and the hinted op never enlarges past the source either
/// way.
pub(in crate::server) const MAX_HINT_WIDTH: u32 = 3840;
const _: () = assert!(MAX_HINT_WIDTH <= crate::url::MAX_DIMENSION);

/// Round a hinted width UP to the next multiple of the
/// quantization step, so a client never gets less than it
/// asked for and the cache sees bounded cardinality, then
/// clamp it to [`MAX_HINT_WIDTH`]. The clamp happens here,
/// before the cache key is built, so hints past the
/// ceiling all share the ceiling's entry. It also keeps the
/// synthesized op inside the parser's dimension cap: the op
/// bypasses parsing, so a near-cap hint at a coarse step
/// must not mint a size the URL grammar itself would
/// reject.
pub(in crate::server) fn quantize_width(width: u32, step: u32) -> u32 {
    let step = step.max(1);
    (width.div_ceil(step) * step).min(MAX_HINT_WIDTH)
}

/// Whether the client sent `Save-Data: on`.
pub(in crate::server) fn save_data_hint(headers: &HeaderMap) -> bool {
    headers
        .get(SAVE_DATA_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("on"))
}

fn generate_request_id() -> String {
    use sha2::Digest as _;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let digest = sha2::Sha256::digest(format!("{nanos}-{n}").as_bytes());
    hex::encode(&digest[..8])
}

/// The request ID a client chose with `X-Request-Id`, when
/// it is 1 to 128 letters, digits, `-` or `_`.
fn client_request_id(inbound: Option<&str>) -> Option<String> {
    let value = inbound?.trim();
    ((1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
    .then(|| value.to_string())
}

/// One image request's record while the request is under way,
/// owed to the request log and the metrics (issue #707).
///
/// Every image request the server handles leaves one line and
/// one metrics update. [`OwedRecord::file`] leaves them with the
/// status of the answer. A request whose client goes away is
/// dropped where it is, and so is its record, which then files
/// itself with [`crate::observe::CLIENT_CLOSED`]: at once when the
/// request stops where it was (a follower waiting on another
/// request's render, a miss whose render has not started, a meta
/// or debug request), and when the render ends for a render that
/// outlives its client and is stored (see [`finish_started_miss`]).
/// The record's own fields say how far the request got; it served
/// no bytes, whatever it rendered.
///
/// It holds the request's handle on the served state, as every
/// task a request starts does (issue #475), not clones of the
/// metrics and the log. Its time runs from `started`, which the
/// handler takes before its wait for room in the request log, so
/// the wait counts. A record dropped while its thread panics files
/// nothing, as a request that panicked left no line before.
pub(in crate::server) struct OwedRecord {
    record: RequestRecord,
    started: Instant,
    /// The state it is owed to: `None` once it is filed, and for
    /// a record owed nowhere.
    state: Option<Arc<AppState>>,
}

impl OwedRecord {
    /// `record`, for a request that started at `started`, owed
    /// to `state`'s metrics and request log.
    pub(in crate::server) fn new(
        record: RequestRecord,
        started: Instant,
        state: Arc<AppState>,
    ) -> Self {
        Self {
            record,
            started,
            state: Some(state),
        }
    }

    /// A record owed nowhere: filing or dropping it records
    /// nothing. For a request whose own record was filed where
    /// it was dropped.
    pub(in crate::server) fn settled() -> Self {
        Self {
            record: RequestRecord::default(),
            started: Instant::now(),
            state: None,
        }
    }

    /// File the record: the request was answered with `status`.
    pub(in crate::server) fn file(mut self, status: u16) {
        self.record.status = status;
        self.file_now();
    }

    /// Record the request in the metrics and log its line, once.
    fn file_now(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        self.record.total = self.started.elapsed();
        state.metrics.record(&self.record);
        state.request_log.line(&self.record.to_json());
    }
}

impl Drop for OwedRecord {
    fn drop(&mut self) {
        if self.state.is_none() || std::thread::panicking() {
            return;
        }
        // The client went away before the answer: nothing was
        // sent.
        self.record.status = crate::observe::CLIENT_CLOSED;
        self.record.bytes_out = 0;
        self.file_now();
    }
}

impl std::ops::Deref for OwedRecord {
    type Target = RequestRecord;

    fn deref(&self) -> &RequestRecord {
        &self.record
    }
}

impl std::ops::DerefMut for OwedRecord {
    fn deref_mut(&mut self) -> &mut RequestRecord {
        &mut self.record
    }
}

pub(in crate::server) async fn handle(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    // The request's time runs from here, before its wait for
    // room in the request log, so its line and the latency
    // metrics count that wait (issue #707).
    let started = Instant::now();
    // Every request here leaves a log line, so it waits here
    // while the request log's queue is full (issue #642):
    // before anything is done for it, so a client that gives
    // up meanwhile has cost nothing and is owed no line.
    state.request_log.room().await;
    // The request's URI and headers are read where they are
    // (issue #641): the `Uri` and `HeaderMap` extractors
    // hand a handler a copy of each. The body goes at once,
    // as it did with them.
    let (parts, body) = request.into_parts();
    drop(body);
    let (uri, headers) = (&parts.uri, &parts.headers);
    // The raw path, exactly as sent: the signature covers it
    // byte-for-byte, so no percent-decoding may happen first
    // (the `Path` extractor decodes, which would rewrite e.g.
    // an encoded watermark URL before verification). At most
    // one leading slash is stripped (audit PL-7): a doubled
    // slash leaves an empty signature segment, which the
    // parser rejects, instead of silently serving the same
    // content under a second spelling.
    let path = uri.path().strip_prefix('/').unwrap_or(uri.path());
    let chosen_id = client_request_id(headers.get("x-request-id").and_then(|v| v.to_str().ok()));
    let chose_id = chosen_id.is_some();
    // From here the request owes its line and its metrics
    // update (issue #707): its record files them when it is
    // answered, below, or when it is dropped, should the
    // client go away first (see `OwedRecord`).
    let record = OwedRecord::new(
        RequestRecord {
            method: if parts.method == axum::http::Method::HEAD {
                "HEAD"
            } else {
                "GET"
            },
            request_id: chosen_id.unwrap_or_else(generate_request_id),
            ..RequestRecord::default()
        },
        started,
        Arc::clone(&state),
    );
    let (mut response, record) = serve(&state, record, path, headers).await;
    // Every answer on this route, errors and meta included,
    // names the request headers it may depend on.
    response
        .headers_mut()
        .insert(header::VARY, image_vary(state.client_hints));
    // An answer a shared cache may store reaches every later
    // viewer of the URL with the headers it was stored with,
    // so it never carries an ID the client chose: that
    // client already has it, and the log line records it.
    // An ID the server generated is still sent, so the
    // answer's caching is read only for one the client
    // chose.
    let shared = chose_id
        && response
            .headers()
            .get(header::CACHE_CONTROL)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| has_directive(v, "public"));
    if !shared {
        response.headers_mut().insert(
            axum::http::HeaderName::from_static("x-request-id"),
            record.request_id.parse().unwrap(),
        );
    }
    // The line goes to the request log (issue #642), which
    // by default queues it for its writer thread: the
    // response does not wait for a `write`.
    record.file(response.status().as_u16());
    response
}

/// Run one image request, timing each stage into its record,
/// which comes back with the answer.
///
/// The request runs in stages (issue #479):
/// [`refuse_own_fetch`], [`authenticate`], [`resolve_request`]
/// and [`negotiate`] admit it, each returning a [`Refusal`]
/// that is answered here, the one place that knows the
/// request's spelling;
/// [`cache_key_for`] keys it; and [`run_flight`] serves it
/// from the tiers or the miss path.
pub(in crate::server) async fn serve(
    state: &Arc<AppState>,
    mut record: OwedRecord,
    path: &str,
    headers: &HeaderMap,
) -> (Response, OwedRecord) {
    let accept = headers.get(header::ACCEPT).and_then(|v| v.to_str().ok());
    let if_none_match = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok());
    // With client hints off, no hint header is read at all:
    // every request for a URL gets the same bytes, the one
    // answer a cache that keys on the URL alone can hold.
    let hints = state.client_hints;
    let dpr = dpr_hint(headers).filter(|_| hints);
    let width = width_hint(headers).filter(|_| hints);
    let save_data = hints && save_data_hint(headers);
    let full = format!("/{path}");
    let opened =
        refuse_own_fetch(state, headers).and_then(|()| authenticate(state, &full, &mut record));
    let sealed = opened
        .as_ref()
        .is_ok_and(|opened| matches!(opened.credential, Credential::Sealed));
    let admitted = opened.and_then(|opened| {
        let resolved = resolve_request(state, &opened.remainder, &mut record)?;
        let negotiated = negotiate(state, &resolved.ops, accept, dpr, width, save_data)?;
        Ok((resolved, negotiated))
    });
    let (resolved, negotiated) = match admitted {
        Ok(admitted) => admitted,
        Err(refusal) => return (refusal.answer(sealed, accept), record),
    };
    let hint_shaped = negotiated.hint_shaped();
    let cache_key = cache_key_for(
        &state.config_fingerprint,
        &resolved.ops,
        &resolved.source_url,
        &negotiated,
    );
    // The key fully determines the lookup: a stored response can
    // be served without touching the loader or the engine. Meta
    // requests bypass the cache in both directions: they must
    // never be answered with image bytes, and their JSON is never
    // stored as an image entry. Debug requests bypass it too:
    // the diagnostic must never be served from, or stored into,
    // any tier (its key differs from the plain URL's anyway, but
    // the bypass is the rule, not the key's accident).
    let bypass = negotiated.meta || negotiated.debug;
    if !bypass {
        record.cache_looked_up = true;
    }
    let work = MissWork {
        ops: resolved.ops,
        source_url: resolved.source_url,
        directives: negotiated.directives,
        preferred: negotiated.preferred,
        meta: negotiated.meta,
        debug: negotiated.debug,
        placeholders: negotiated.placeholders,
        hinted_width: negotiated.hinted_width,
        dpr_hint: dpr,
        // The engine steps on the same decision the key and
        // the private marking take, never on the raw header:
        // a render the header stepped where the key does not
        // separate it would be stored, and served publicly,
        // as the URL's plain answer.
        save_data: negotiated.save_data_applies,
        accept: accept.map(str::to_string),
        groups: resolved.groups,
        cache_key,
        max_age: resolved.max_age,
        expires_at: resolved.expiry.map(|expiry| expiry.at),
        sealed,
        progress: MissProgress::default(),
        disk_handoff: None,
        loop_entries: crate::loader::LoopEntries::of(headers),
    };
    let (mut response, record) = run_flight(state, path, work, if_none_match, record).await;
    if hint_shaped {
        keep_from_shared_caches(&mut response);
    }
    (response, record)
}

/// Make a public answer private, keeping its lifetime: an
/// answer a client hint gives its own cache entry (see
/// [`Negotiated::hint_shaped`]).
///
/// The hint headers are request headers any client sets,
/// and many shared caches (CDNs among them) key on the URL
/// and not on `Vary`, so a hint-shaped answer stored there
/// would be the answer every later viewer of the URL gets,
/// whatever their own headers say. A private answer is
/// still kept by the browser that asked, for its whole
/// lifetime, and the URL's hint-free answer stays public:
/// it is the answer a cache that keys on the URL alone
/// should hold. Answers that are not public (meta, errors,
/// the fallback, the debug view) are left as they are.
fn keep_from_shared_caches(response: &mut Response) {
    let Some(rest) = response
        .headers()
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("public"))
    else {
        return;
    };
    let private = format!("private{rest}");
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, private.parse().unwrap());
}

/// The fifth stage of [`serve`]: the request's cache key.
///
/// Without an explicit format in the URL, the response depends
/// on the Accept class as well as the path, so the cache key
/// carries the class. Explicit-format URLs key on the path alone.
/// The key is the request's canonical form (Q1): the
/// parsed operations re-serialized in their one
/// spelling, so equivalent requests share an entry.
/// Expiry and the cache lifetime are not operations,
/// so neither ever enters it. A form past
/// [`crate::cache::MAX_KEY_BASE_BYTES`] is carried as its
/// digest ([`crate::cache::bounded_key_base`]), so a padded
/// source cannot make one entry's key, which every tier
/// stores and the cache maps hold twice, tens of kilobytes
/// long.
pub(in crate::server) fn cache_key_for(
    config_fingerprint: &str,
    ops: &[crate::url::Operation],
    source_url: &str,
    negotiated: &Negotiated,
) -> String {
    use std::fmt::Write as _;
    let key_base =
        crate::cache::bounded_key_base(crate::url::canonical_form(ops, source_url), source_url);
    // The key also carries the configuration fingerprint
    // (audit R2-08): entries in the persistent tiers are
    // keyed under the configuration that produced them, so
    // a restart with different output-shaping settings
    // (metadata policy, quality tuning, width step, model
    // or font or encoder files) starts a fresh epoch
    // instead of serving the old configuration's renders.
    // The fingerprint sits inside the base, ahead of the
    // negotiated-class and hint suffixes. Each part is
    // appended to the one buffer (issue #640): the key is
    // built for every request.
    let mut cache_key = key_base;
    cache_key.reserve(config_fingerprint.len() + 24);
    cache_key.push_str("#cfg");
    cache_key.push_str(config_fingerprint);
    if negotiated.directives.explicit_format.is_none() {
        cache_key.push_str(match negotiated.preferred {
            Some(ImageFormat::Avif) => "#avif",
            Some(ImageFormat::Jxl) => "#jxl",
            Some(ImageFormat::WebP) => "#webp",
            _ => "#source",
        });
    }
    if let Some(w) = negotiated.hinted_width {
        let _ = write!(cache_key, "#w{w}");
    }
    if negotiated.save_data_applies {
        cache_key.push_str("#sd");
    }
    let effective_dpr = negotiated.effective_dpr;
    if effective_dpr != 1.0 {
        let _ = write!(cache_key, "#dpr{effective_dpr}");
    }
    cache_key
}
