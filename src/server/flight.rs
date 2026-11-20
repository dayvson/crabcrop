use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::miss::{MissWork, serve_miss};
use super::request::OwedRecord;
use super::response::{attach_cache_headers, finalize_validators, respond};
use super::tiers::{DiskBudget, DiskHandoff, finish_disk_write, lookup_tiers_within};
use super::{
    AppState, Flight, FlightSlot, LeaderGuard, MetaFlight, MissProgress, SharedFlight,
    SharedResponse, wants_detections, wants_embeddings, wants_moderation, wants_ocr,
};
use crate::cache::CachedImage;
use crate::url::ImageFormat;

#[cfg(doc)]
use super::Background;
#[cfg(doc)]
use super::request::serve;
#[cfg(doc)]
use super::tiers::populate_tiers;

/// The flight key of a meta request (issue #295): the
/// cache key plus every fact the document depends on
/// beyond it (the sealed spelling, which redacts the
/// document, the placeholder fields, and the Accept
/// opt-ins), each computed by the same helpers the miss
/// path branches on, so the key can never disagree with
/// the document a flight shares.
fn meta_flight_key(work: &MissWork) -> String {
    let cache_key = &work.cache_key;
    let accept = work.accept.as_deref();
    format!(
        "{cache_key}\u{1}meta\u{1}{}{}{}{}{}{}",
        work.sealed as u8,
        work.placeholders as u8,
        wants_detections(accept) as u8,
        wants_moderation(accept) as u8,
        wants_embeddings(accept) as u8,
        wants_ocr(accept) as u8,
    )
}

/// The last stage of [`serve`]: serve the keyed request.
/// A debug request runs the miss path on its own, and a
/// meta request coalesces through its own flights
/// ([`serve_meta_flight`]).
///
/// An image request reads the tiers and coalesces its miss
/// through its key's flight (audit R2-09): the first miss
/// for a key leads (one fetch, one encode, under the miss
/// path's gates); concurrent requests for the same key wait
/// for the leader, then re-check the tiers and serve its
/// entry. A leader whose result no tier stored (an error, a
/// fallback, a render with the caches off) leaves its
/// response in the flight's slot instead, and a waiter
/// whose request is identical to the leader's answers with
/// it (issue #407): it is the answer the waiter would have
/// produced itself, and taking it keeps a burst on a
/// failing origin at one fetch per flight rather than one
/// per waiter. Identical means the same [`FlightIdentity`].
/// The conditional header is left out of it: each waiter
/// applies its own. A leader whose client goes away after
/// its render has started still finishes the flight (see
/// [`finish_started_miss`]). A waiter that cannot take the
/// response (another spelling of the key, a body past the
/// share cap, a leader whose request was dropped before its
/// render started, a leader that forwent its fallback run
/// because its client had gone, a run a purge of the key
/// overtook) leads or follows the next flight. Waits are
/// capped: a waiter whose flights keep ending without an
/// answer for it stops queueing and runs the miss path
/// itself (issue #375, [`own_miss`]), so it ends with the
/// origin's real answer instead of a busy response masking
/// it.
///
/// Each pass of the loop reads the tiers ([`answer_hit`]),
/// takes what the flight it followed left for it
/// ([`Following::take_shared`], [`answer_shared`]), and
/// then leads the key's flight ([`lead_flight`]) or follows
/// it ([`Following::wait`]). The loop stays in this function
/// and the misses are awaited here: an async function holds
/// its arguments for as long as it runs, so a stage of its
/// own that takes the work would hold a second copy of it
/// in every request's future.
pub(in crate::server) async fn run_flight(
    state: &Arc<AppState>,
    path: &str,
    work: MissWork,
    if_none_match: Option<&str>,
    record: OwedRecord,
) -> (Response, OwedRecord) {
    // The miss path is boxed at each of the four places this
    // function reaches it (issue #638): the two below, and in
    // the futures [`lead_flight`] and [`own_miss`] build. Its
    // future holds the loader's and the engine runner's
    // state, about 20 KB. Awaited in line it made every
    // request's future that large, and a request allocates
    // and moves its future once, whether it goes on to miss
    // or not.
    // Debug requests bypass everything, per request.
    if work.debug {
        let (response, record, _) = Box::pin(serve_miss(state, work, record)).await;
        return (response, record);
    }
    // Meta requests coalesce through their own flights
    // (issue #295), keyed by [`meta_flight_key`].
    if work.meta {
        let flight_key = meta_flight_key(&work);
        return Box::pin(serve_meta_flight(state, work, record, &flight_key)).await;
    }
    let identity = FlightIdentity::of(path, &work);
    let mut following = Following::default();
    // One disk budget for every tier check this request makes
    // (issue #572).
    let mut budget = DiskBudget::new();
    loop {
        // The key and the groups are read where they are
        // until the miss path takes the work.
        let (cache_key, groups) = (work.cache_key.as_str(), work.groups.as_slice());
        if let Some((hit, cache_status)) =
            lookup_tiers_within(state, cache_key, groups, &mut budget).await
        {
            return answer_hit(state, &work, hit, cache_status, if_none_match, record);
        }
        if let Some(shared) = following.take_shared(state, cache_key, groups) {
            return answer_shared(&shared, if_none_match, record);
        }
        match state.inflight.lead(cache_key) {
            Flight::Leader(guard, slot) => {
                let request = following.request.take().unwrap_or_else(|| identity.text());
                let (progress, lead) = lead_flight(state, work, guard, slot, request, record);
                let answer = finish_started_miss(state, progress, lead).await;
                return answer_miss(answer, if_none_match);
            }
            Flight::Follower(changed, slot) => {
                if following.waits >= MAX_FLIGHT_WAITS {
                    let (progress, miss) = own_miss(state, work, record);
                    let answer = finish_started_miss(state, progress, miss).await;
                    return answer_miss(answer, if_none_match);
                }
                following.wait(changed, slot, &identity).await;
            }
        }
    }
}

/// What makes two image requests for one key identical for
/// a flight's shared response (issue #407): the same raw
/// path (the sealing, the cache groups, and the cache
/// lifetime all come from it) and the same header inputs as
/// the miss path sees them: the negotiated format, not the
/// raw Accept (on this path errors are plain text whatever it
/// says), the quantized width, not the raw hint, and whether
/// Save-Data applies, not the raw header, so browsers that
/// phrase the same request differently still share one
/// flight.
struct FlightIdentity<'a> {
    path: &'a str,
    preferred: Option<ImageFormat>,
    dpr_hint: Option<f32>,
    hinted_width: Option<u32>,
    save_data: bool,
}

impl<'a> FlightIdentity<'a> {
    fn of(path: &'a str, work: &MissWork) -> Self {
        Self {
            path,
            preferred: work.preferred,
            dpr_hint: work.dpr_hint,
            hinted_width: work.hinted_width,
            save_data: work.save_data,
        }
    }

    /// The identity as a flight's slot records it
    /// ([`SharedFlight::request`]).
    ///
    /// It is built only by a request that leads or follows a
    /// flight (issue #641): a hit compares it with nothing. A
    /// follower builds it before it waits, and a leader once
    /// it holds the flight, never between a lookup that missed
    /// and the claim. Whatever a request does there widens the
    /// window in which the flight it would have followed can
    /// end unseen, and the request then leads a second flight
    /// for a render the first has just stored.
    fn text(&self) -> String {
        let Self {
            path,
            preferred,
            dpr_hint,
            hinted_width,
            save_data,
        } = self;
        format!("{path:?}{preferred:?}{dpr_hint:?}{hinted_width:?}{save_data}")
    }
}

/// An image request's part in the flights it has followed.
#[derive(Default)]
struct Following {
    /// Its [`FlightIdentity`], once it has built it.
    request: Option<String>,
    /// How many flights it has waited on.
    waits: u32,
    /// The slot of the flight it waited on last, until the
    /// next pass reads it.
    slot: Option<Arc<FlightSlot>>,
}

impl Following {
    /// The response the flight this request waited on last
    /// left in its slot, when the request is identical to
    /// the leader's and no purge covering the key has been
    /// recorded since the leader's run began.
    fn take_shared(
        &mut self,
        state: &AppState,
        cache_key: &str,
        groups: &[String],
    ) -> Option<SharedFlight> {
        self.slot
            .take()
            .and_then(|slot| {
                slot.shared
                    .lock()
                    .unwrap()
                    .as_ref()
                    .filter(|shared| Some(&shared.request) == self.request.as_ref())
                    .cloned()
            })
            .filter(|shared| {
                !state
                    .purge_guard
                    .purge_covers(shared.snapshot, cache_key, groups)
            })
    }

    /// Wait for the flight this request follows to end.
    async fn wait(
        &mut self,
        mut changed: tokio::sync::watch::Receiver<bool>,
        slot: Arc<FlightSlot>,
        identity: &FlightIdentity<'_>,
    ) {
        self.waits += 1;
        self.request.get_or_insert_with(|| identity.text());
        // The leader always signals on drop; the timeout is
        // only a backstop against a wedged leader, after which
        // the waiter re-checks the tiers and may lead itself.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(60), changed.changed()).await;
        self.slot = Some(slot);
    }
}

/// The answer to an image request from a tier hit.
fn answer_hit(
    state: &AppState,
    work: &MissWork,
    hit: CachedImage,
    cache_status: &'static str,
    if_none_match: Option<&str>,
    mut record: OwedRecord,
) -> (Response, OwedRecord) {
    record.cache_hit = true;
    record.bytes_out = hit.bytes.len();
    let mut response = respond(
        hit.format,
        hit.quality,
        hit.bytes,
        work.max_age,
        state.client_hints,
    );
    attach_cache_headers(&mut response, cache_status, &work.groups, work.sealed);
    finalize_validators(&mut response, &hit.etag, if_none_match);
    (response, record)
}

/// The answer to an image request from the response the
/// flight it followed shared (issue #407).
fn answer_shared(
    shared: &SharedFlight,
    if_none_match: Option<&str>,
    mut record: OwedRecord,
) -> (Response, OwedRecord) {
    record.bytes_out = shared.response.body.len();
    record.fallback = shared.fallback;
    let mut response = respond_shared(&shared.response);
    if let Some(etag) = &shared.etag {
        finalize_validators(&mut response, etag, if_none_match);
    }
    (response, record)
}

/// The leader's part of the key's flight, as a miss to run
/// (see [`finish_started_miss`]) with the progress it
/// reports: run the miss, leave its response in the
/// flight's slot for the followers (`request` is this
/// request's [`FlightIdentity`]), wake them by dropping
/// `guard`, and then finish the render's disk write.
///
/// The leader's whole part runs as one miss: a render it
/// starts is stored and answers the followers even when
/// this request is gone, and the guard is held until it
/// has. A miss that forwent a render for a client that had
/// gone ends without its request's answer, so it shares
/// nothing.
fn lead_flight(
    state: &Arc<AppState>,
    mut work: MissWork,
    guard: LeaderGuard,
    slot: Arc<FlightSlot>,
    request: String,
    record: OwedRecord,
) -> (
    MissProgress,
    impl std::future::Future<Output = (Response, OwedRecord, Option<String>)> + Send + 'static,
) {
    let snapshot = state.purge_guard.snapshot();
    let progress = work.progress.clone();
    let task_progress = progress.clone();
    let task_state = state.clone();
    // The render's disk write waits until the followers are
    // woken (issue #572): they read the render from memory
    // and never wait for it.
    let handoff = DiskHandoff::default();
    work.disk_handoff = Some(handoff.clone());
    let lead = async move {
        let (response, record, render_etag) = Box::pin(serve_miss(&task_state, work, record)).await;
        let (response, shared) = share_response(response, FLIGHT_SHARE_CAP).await;
        if let Some(shared) = shared.filter(|_| !task_progress.forwent_render()) {
            *slot.shared.lock().unwrap() = Some(SharedFlight {
                request,
                snapshot,
                response: shared,
                etag: render_etag.clone(),
                fallback: record.fallback,
            });
        }
        drop(guard);
        let pending = handoff.lock().unwrap().take();
        if let Some(pending) = pending {
            finish_disk_write(
                &task_state,
                pending.snapshot,
                &pending.key,
                &pending.entry,
                pending.claim,
                false,
            )
            .await;
        }
        (response, record, render_etag)
    };
    (progress, lead)
}

/// The miss of a request whose flight waits are spent, to
/// run (see [`finish_started_miss`]) with the progress it
/// reports: the same settlement the meta flights use (issue
/// #375). Shedding a busy response here masked the origin's
/// real answer (its error, or a slow success) behind a
/// server_busy this request never earned: the cap exists to
/// stop queueing behind failing leaders forever, not to
/// refuse a request that has not had its own attempt.
fn own_miss(
    state: &Arc<AppState>,
    work: MissWork,
    record: OwedRecord,
) -> (
    MissProgress,
    impl std::future::Future<Output = (Response, OwedRecord, Option<String>)> + Send + 'static,
) {
    let progress = work.progress.clone();
    let task_state = state.clone();
    let miss = async move { Box::pin(serve_miss(&task_state, work, record)).await };
    (progress, miss)
}

/// A miss's answer to the request that ran it: the render's
/// ETag answers the request's conditional headers.
fn answer_miss(
    (mut response, record, render_etag): (Response, OwedRecord, Option<String>),
    if_none_match: Option<&str>,
) -> (Response, OwedRecord) {
    if let Some(etag) = &render_etag {
        finalize_validators(&mut response, etag, if_none_match);
    }
    (response, record)
}

/// Run one image miss (`miss`: the miss path for a key,
/// with whatever its flight does around it) on a task of
/// its own, tracked for the shutdown drain, and wait for
/// its answer.
///
/// The task is what lets a render outlive its client
/// usefully. When the client goes away, the server drops
/// this future, and with it the sender of the `gone`
/// channel. If the miss has not started a render by then
/// (`progress`, see [`MissProgress`]), the task drops the
/// miss too, which stops it where it was, as before: a
/// source load or a gate wait ends, its budget share and
/// permits come back, and it starts no render. If a render
/// has started, it runs to its end on its blocking thread
/// anyway, so the task finishes the miss: the render is
/// stored in the tiers (the guarded writes of
/// [`populate_tiers`], unchanged) and the flight's
/// followers receive it, through the tiers or the flight's
/// slot. A replay of the request is then a hit, or a
/// follower of the render still running, instead of a
/// fresh render each time. Finishing starts no further
/// render: a source that did not decode gets no fallback
/// run once the client has gone, and the flight's
/// followers then make their own attempt. The permits a
/// render holds still come back when that render ends
/// (issue #357), whether or not its client is still there;
/// what runs after it (the tier writes, the slot) holds
/// none. Only image misses run here: meta documents and
/// debug views are never stored, so a render made for one
/// has nothing to keep once its client is gone.
///
/// The request's record travels with the miss, so a miss
/// that is dropped files its record as one whose client
/// left, and one that finishes for nobody files it when
/// the render and its tier writes are done, with the load
/// and render times they took (issue #707).
///
/// A miss whose client is still there is part of a
/// request in flight, which the shutdown drain counts as
/// such; one whose client has gone is counted as a render
/// left running (see [`Background::spawn_miss`]).
pub(in crate::server) async fn finish_started_miss<F>(
    state: &AppState,
    progress: MissProgress,
    miss: F,
) -> (Response, OwedRecord, Option<String>)
where
    F: std::future::Future<Output = (Response, OwedRecord, Option<String>)> + Send + 'static,
{
    // Never sent on: dropping the sender, with this
    // future, is the signal.
    let (_waiting, gone) = tokio::sync::oneshot::channel::<std::convert::Infallible>();
    let background = state.background.clone();
    let task = state.background.spawn_miss(async move {
        let mut miss = std::pin::pin!(miss);
        tokio::select! {
            // The client's departure is checked first on
            // every wake, so a miss whose client has gone
            // never takes a lane permit that frees up in
            // the same instant and starts a render.
            biased;
            _ = gone => {
                if progress.render_started() {
                    progress.mark_client_gone();
                    let _left = background.client_left();
                    // The render finishes for nobody. Its
                    // request's record files itself here, as
                    // one whose client left, while the drain
                    // still counts the render (issue #707).
                    let (_, record, _) = miss.await;
                    drop(record);
                }
                // Otherwise the miss drops where it is, and
                // its record with it, which files the same
                // way.
                None
            }
            answer = &mut miss => Some(answer),
        }
    });
    match task.await {
        Ok(Some(answer)) => answer,
        // A panic on the miss path stays the panic of
        // this request, as when the path ran inline.
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        // The task gives up only once this future is gone,
        // and is cancelled only by a runtime that is
        // shutting down, which drops this request too: no
        // client receives this answer. The request's record
        // went with the task, and filed itself there.
        _ => (
            StatusCode::SERVICE_UNAVAILABLE.into_response(),
            OwedRecord::settled(),
            None,
        ),
    }
}

/// The body size above which an image flight's response
/// is not shared with its followers (issue #407). The
/// body is already in memory and followers share the
/// leader's buffer rather than copying it, so the cap
/// only bounds what one flight can pin for its followers;
/// it sits well above any realistic still render.
const FLIGHT_SHARE_CAP: usize = 64 * 1024 * 1024;

/// The body size above which a meta document is not
/// shared with a flight's followers (issue #295): real
/// documents are KB-scale, so the cap only ever refuses
/// the pathological.
const META_SHARE_CAP: usize = 4 * 1024 * 1024;

/// Meta singleflight (issue #295): concurrent identical
/// meta requests share one run of the miss path instead
/// of each loading and rendering the same source. The
/// leader runs `serve_miss` unchanged and collects its
/// response into the flight's slot before its guard
/// drops, and followers answer with a reconstructed
/// response (the leader's status, headers, and body) and
/// their own request record, which truthfully shows they
/// ran no load and no process stage. Errors are shared
/// too (issue #407): meta errors are always the JSON
/// envelope, shaped only by inputs the flight key pins,
/// so a follower receives exactly the failure a solo
/// request gets, and a burst against a failing origin
/// costs one fetch instead of one per follower. A flight
/// that leaves its slot empty (a body past the share
/// cap, a leader whose request was dropped) sends its
/// followers to lead or follow the next flight, and
/// finally to run the miss path themselves. Nothing
/// outlives the flight: meta stays uncached across
/// time.
///
/// The two choices this settles, deliberately: request
/// records stay per request with no flight marker (the
/// record type has no such field, and none is invented;
/// a follower is recognizable by its zero load and
/// process times), and a follower's response headers are
/// the leader's, which is sound because the flight key
/// pins every input the headers and the document were
/// computed from (see [`meta_flight_key`]).
async fn serve_meta_flight(
    state: &AppState,
    work: MissWork,
    mut record: OwedRecord,
    key: &str,
) -> (Response, OwedRecord) {
    let mut waits = 0;
    loop {
        match state.inflight.lead_meta(key) {
            MetaFlight::Leader(guard, slot) => {
                let (response, record, _) = serve_miss(state, work, record).await;
                let (response, shared) = share_response(response, META_SHARE_CAP).await;
                if let Some(shared) = shared {
                    *slot.shared.lock().unwrap() = Some(shared);
                }
                drop(guard);
                return (response, record);
            }
            MetaFlight::Follower(mut changed, slot) => {
                if waits >= MAX_FLIGHT_WAITS {
                    let (response, record, _) = serve_miss(state, work, record).await;
                    return (response, record);
                }
                waits += 1;
                // The leader always signals on drop; the
                // timeout is only a backstop against a
                // wedged leader, after which the waiter
                // re-checks the slot and may lead itself.
                let _ = tokio::time::timeout(std::time::Duration::from_secs(60), changed.changed())
                    .await;
                if let Some(shared) = slot.shared.lock().unwrap().clone() {
                    record.bytes_out = shared.body.len();
                    return (respond_shared(&shared), record);
                }
            }
        }
    }
}

/// Rebuild a response from a flight's shared response:
/// the leader's status and headers, its body bytes.
fn respond_shared(shared: &SharedResponse) -> Response {
    let mut response = Response::new(axum::body::Body::from(shared.body.clone()));
    *response.status_mut() = shared.status;
    *response.headers_mut() = shared.headers.clone();
    response
}

/// Split a leader's response into the shared form a
/// flight slot carries, rebuilding the leader's own
/// response from the same parts. Returns `None` for the
/// shared form when the body is not collected whole
/// within `cap`; the miss path's bodies are in-memory,
/// so in practice only the cap refuses, and the leader's
/// response is rebuilt untouched in that case whenever
/// the body was not consumed trying.
async fn share_response(response: Response, cap: usize) -> (Response, Option<SharedResponse>) {
    let (parts, body) = response.into_parts();
    if <axum::body::Body as axum::body::HttpBody>::size_hint(&body)
        .exact()
        .is_none_or(|n| n > cap as u64)
    {
        return (Response::from_parts(parts, body), None);
    }
    match axum::body::to_bytes(body, cap).await {
        Ok(bytes) => {
            let shared = SharedResponse {
                status: parts.status,
                headers: parts.headers.clone(),
                body: bytes,
            };
            let response = Response::from_parts(parts, axum::body::Body::from(shared.body.clone()));
            (response, Some(shared))
        }
        Err(_) => {
            // Unreachable for the in-memory bodies the
            // miss path answers with (the size hint above
            // already vouched for the body); the status
            // and headers survive, the body does not.
            stderr_line!("flight: leader body could not be collected for sharing");
            (Response::from_parts(parts, axum::body::Body::empty()), None)
        }
    }
}

/// How many times one request may wait on another
/// request's in-flight miss for the same key before it
/// stops waiting and runs the miss path itself (audit
/// R2-09, issue #375). A burst of identical requests
/// settles on the first flight (issue #407), so the
/// cap only comes into play for waiters no flight could
/// answer.
pub(in crate::server) const MAX_FLIGHT_WAITS: u32 = 3;
