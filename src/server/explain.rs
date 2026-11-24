//! `/explain` (issue #741): the route's handler and its
//! stages. It admits through the image path's own stages,
//! probes the source within the source buffer budget and
//! under the probe gate, and answers with the report as JSON
//! or HTML; the report itself is built in `crate::explain`.
//!
//! server.rs re-exports what the router and the tests use.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

use super::admission::{
    Credential, Opened, Refusal, Resolved, authenticate, refuse_own_fetch, resolve_request,
    unix_seconds,
};
use super::request::dpr_hint;
use super::response::{
    accept_of, bad_signature_parts, busy_response, error_response, load_busy_response,
    load_error_parts,
};
use super::{AppState, PROCESS_GATE_WAIT, SOURCE_BUDGET_WAIT};
use crate::loader::{LoadError, SourceCharge};
use crate::observe::RequestRecord;

#[cfg(doc)]
use super::miss::render_miss;

/// Whether a probe head settles the explain report's
/// source facts: the probe knows the kind, and it either
/// has the dimensions or the kind is one whose dimensions
/// never live in headers (video, PDF). Anything else means
/// the identifying header may still be on its way.
///
/// The load asks this after every chunk, on the async
/// runtime, so it never sizes a kind whose probe is heavy
/// ([`crate::engine::probe_is_heavy`]): an SVG's size needs
/// the whole document anyway, an AVIF's a decode, a HEIC's a
/// walk of its item graph, and a JPEG XL's the decoder's
/// reading of its colour encoding. Those read on to the end
/// of the body (or the head cap), and [`explain_probe`]
/// sizes them once, off the runtime.
pub(in crate::server) fn probe_settled(head: &[u8]) -> bool {
    let kind = crate::engine::probe_kind(head);
    match kind {
        "unknown" => false,
        "video" | "pdf" => true,
        _ if crate::engine::probe_is_heavy(kind) => false,
        _ => crate::engine::probe_source(head).dimensions.is_some(),
    }
}

/// How long an explain probe may wait for a probe slot
/// before the server sheds it with a 503: the stills
/// gate's wait.
const PROBE_GATE_WAIT: std::time::Duration = PROCESS_GATE_WAIT;

/// The explain report's source probe. A heavy probe (an
/// SVG's parse in the worker, which may take the probe's
/// whole 5 s, an AVIF's decode, a HEIC's item graph walk, or
/// a JPEG XL's header read through its colour encoding) runs
/// on a blocking thread under the probe gate, so the async
/// runtime, and every route it serves, never waits on one;
/// the permit moves into the job, so a probe whose request
/// went away still counts until it ends.
/// Every other kind's probe reads headers alone and runs
/// inline. `charge` holds the head against the source
/// buffer budget and goes where the head goes, so the head
/// stays charged until the probe is done with it, its wait
/// for a probe slot included. `None` means the gate stayed
/// full past its wait.
async fn explain_probe(
    state: &AppState,
    head: Vec<u8>,
    charge: SourceCharge,
) -> Option<crate::engine::SourceProbe> {
    let kind = crate::engine::probe_kind(&head);
    if !crate::engine::probe_is_heavy(kind) {
        return Some(crate::engine::probe_source(&head));
    }
    let permit = tokio::time::timeout(PROBE_GATE_WAIT, state.probe_gate.clone().acquire_owned())
        .await
        .ok()?
        .ok()?;
    let probe = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _charge = charge;
        crate::engine::probe_source(&head)
    })
    .await;
    // A probe that panicked states the kind and no size, as
    // one that failed does.
    Some(probe.unwrap_or(crate::engine::SourceProbe {
        kind,
        dimensions: None,
    }))
}

/// `GET /explain/<signature>/<operations...>/<image-url>`:
/// the self-explaining URL report (issue #196; the report
/// itself is built in [`crate::explain`]).
///
/// The request runs in stages. [`admit_explained`] admits
/// it through the image path's own [`refuse_own_fetch`],
/// [`authenticate`] and [`resolve_request`], so a URL this
/// route admits is one the image route admits, and a
/// refusal is the image path's own status and body (issue
/// #741). Only then is the source touched, once, through
/// the loader with all of its guards and within the source
/// buffer budget ([`probe_explained`]), and the facts come
/// from header probing (`crate::engine::probe_source`): no
/// decode, no detection, no process, AI, or video gate (an
/// SVG, AVIF, HEIC, or JPEG XL probe takes the probe gate, see
/// [`explain_probe`]), and nothing written to any cache
/// tier. A bad signature, an over-cap operation list, or
/// an expired URL therefore costs zero source fetches.
/// [`explain_report`] builds the report and
/// [`explain_response`] answers it.
pub(in crate::server) async fn explain_handle(
    State(state): State<Arc<AppState>>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    let accept = accept_of(&headers);
    let accept = accept.as_deref();
    // With `CRABCROP_EXPLAIN=0` the route is off (issue #591):
    // every request answers the router's own `404 not_found`
    // before the signature is read, so the answer is the
    // same whatever the URL, and no source is fetched.
    if !state.explain_enabled {
        return error_response(
            StatusCode::NOT_FOUND,
            "not_found",
            "not found".to_string(),
            accept,
        );
    }
    let Some(tail) = uri.path().strip_prefix("/explain/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /explain/<signature>/<target>".to_string(),
            accept,
        );
    };
    let full = format!("/{tail}");
    let (credential, resolved) = match admit_explained(&state, &full, &headers) {
        Ok((opened, resolved)) => (opened.credential, resolved),
        // Only plain spellings are admitted here, so every
        // refusal is the plain spelling's own answer, as on
        // the image route.
        Err(refusal) => return refusal.answer(false, accept),
    };
    let entries = crate::loader::LoopEntries::of(&headers);
    let (source_bytes, probe) = match probe_explained(&state, &resolved, &entries, accept).await {
        Ok(facts) => facts,
        Err(response) => return *response,
    };
    let report = explain_report(
        &state,
        &headers,
        credential,
        resolved,
        source_bytes,
        probe,
        full,
    );
    explain_response(&report, accept)
}

/// The first stage of [`explain_handle`]: the image path's
/// admission, [`refuse_own_fetch`], [`authenticate`] and
/// [`resolve_request`] themselves (issue #741), so the two
/// routes cannot disagree on which plain URLs they admit or
/// on how they refuse the rest.
///
/// One of the server's own fetches come back to it is
/// refused first, as on the image route. A sealed spelling
/// is refused before the other two run,
/// wholesale (issue #255): the server never unseals as a
/// service, because an endpoint that opens tokens would be
/// an unsealing oracle for anyone who lifts one off a
/// public page. Operators unseal locally (the CLI's
/// `unseal` verb, under their own secret) and then explain
/// the plain form. The refusal is the uniform
/// `bad_signature`, and no source fetch happens for it.
pub(in crate::server) fn admit_explained<'a>(
    state: &AppState,
    full: &'a str,
    headers: &HeaderMap,
) -> Result<(Opened<'a>, Resolved), Refusal> {
    refuse_own_fetch(state, headers)?;
    let nolead = full.strip_prefix('/').unwrap_or(full);
    if crate::url::looks_sealed(nolead.split('/').next().unwrap_or("")) {
        return Err(Refusal::AsIs(bad_signature_parts()));
    }
    // The stages time their work into a request record. An
    // explain request owes the request log no line, so this
    // one goes nowhere.
    let mut record = RequestRecord::default();
    let opened = authenticate(state, full, &mut record)?;
    let resolved = resolve_request(state, &opened.remainder, &mut record)?;
    Ok((opened, resolved))
}

/// The second stage of [`explain_handle`]: the single
/// bounded source fetch, through the loader and its
/// policy, carrying the request's `CDN-Loop` `entries`,
/// its head charged against the source buffer budget, and
/// the probe of the head it read
/// ([`explain_probe`]). Returns the source's encoded size
/// when the fetch states it, and the probe; a failed fetch
/// (one shed for budget included), or a probe shed for a
/// full gate, is the answer instead, boxed as
/// [`render_miss`] boxes its own (a bare `Response` is 128
/// bytes, too large for an `Err`: clippy's
/// `result_large_err`).
async fn probe_explained(
    state: &AppState,
    resolved: &Resolved,
    entries: &crate::loader::LoopEntries,
    accept: Option<&str>,
) -> Result<(Option<usize>, crate::engine::SourceProbe), Box<Response>> {
    // Watermark and overlay sources are operations' inputs,
    // not the explained URL's source, and are never fetched:
    // no fan-out. A `frame:` segment makes the source a
    // video, so the loader may accept a `video/*`
    // Content-Type for it (issue #214); a `page:` segment
    // does the same for `application/pdf` (issue #215).
    // Explain never runs either tool: the fetch is admitted,
    // the probe stays the raw-bytes probe. Both are read
    // from the merged list, the one the image request runs.
    let admission = crate::loader::MediaAdmission {
        video: resolved
            .ops
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Frame(_))),
        pdf: resolved
            .ops
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Page(_))),
    };
    // The report reads the source's headers only, so the
    // fetch is a probe load: it stops as soon as the head
    // identifies the source instead of downloading the
    // whole body (which also keeps this route off the
    // result cache's coalescing path by construction: it
    // never held a whole body to share). A root source is
    // fetched at its expansion. The head is charged to the
    // source buffer budget as it arrives, as a miss charges
    // its source (issue #403), so a burst of probes waits
    // for budget and sheds like a burst of misses, and a
    // probe an origin keeps reading holds what it received.
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let crate::loader::ProbeLoad { head, total_bytes } = state
        .loader
        .load_probe_charged(
            &resolved.source_url,
            admission,
            &probe_settled,
            &charge,
            entries,
        )
        .await
        .map_err(|e| Box::new(explain_load_error(e, accept)))?;
    let Some(probe) = explain_probe(state, head, charge).await else {
        return Err(Box::new(busy_response(
            "server busy: too many source probes running at once",
            5,
            accept,
        )));
    };
    Ok((total_bytes.map(|n| n as usize), probe))
}

/// The third stage of [`explain_handle`]: the report on an
/// admitted URL and its probed source.
///
/// The forecast, the canonical form and the admission all
/// follow the operations the image request runs, the
/// server-wide defaults merged in (issue #468); the
/// report's operation list is the URL's own, so it never
/// credits the URL with a default. The source is reported
/// exactly as written, with a root's name, never the base
/// URL a root reference resolves to (operator
/// configuration, issue #248).
fn explain_report(
    state: &AppState,
    headers: &HeaderMap,
    credential: Credential,
    resolved: Resolved,
    source_bytes: Option<usize>,
    probe: crate::engine::SourceProbe,
    image_path: String,
) -> crate::explain::Report {
    let ops = resolved.own_ops().to_vec();
    // Each referenced preset with the operations its
    // definition expands to. Definitions were validated at
    // boot; one that no longer parses (it cannot) is listed
    // by name alone rather than failing the report.
    let presets = resolved
        .groups
        .iter()
        .map(|name| {
            let entry_ops = state
                .presets
                .get(name)
                .and_then(|definition| {
                    crate::url::parse_default_ops(definition.as_str(), &state.presets).ok()
                })
                .unwrap_or_default();
            crate::explain::preset_entry(name, &entry_ops)
        })
        .collect();
    // The canonical form is the cache key's path form, of
    // the merged list as the image path keys it. The key's
    // config fingerprint suffix is operator state and stays
    // unreported. A form past the key-length bound is shown
    // in full, though the key carries its digest
    // (`cache::bounded_key_base`). For a root source the
    // form is built from the source as written: the key
    // itself is keyed on the expansion, but printing that
    // here would publish the root's base URL into the
    // report.
    let source_url = resolved.written_source();
    let canonical = crate::url::canonical_form(&resolved.ops, &source_url);
    let (signed, key_id) = match credential {
        Credential::Signature(key_id) => (true, key_id),
        Credential::Unsafe => (false, None),
        // Never admitted here (see `admit_explained`).
        Credential::Sealed => (true, None),
    };
    // The expiry verdict reads the clock the admission read,
    // so the report never calls a URL it admitted expired.
    let (expiry, now) = match resolved.expiry {
        Some(expiry) => (Some(expiry.at), expiry.checked_at),
        None => (None, unix_seconds()),
    };
    crate::explain::Report::new(crate::explain::ReportInput {
        signed,
        key_id,
        expiry,
        now,
        source_url,
        source_root: resolved.root.map(|r| r.name),
        probe,
        // The DPR hint is read as the image path reads it, so
        // the forecast steps a hinted factor down as the image
        // request does (issue #529), and not at all with
        // client hints off.
        dpr_hint: dpr_hint(headers).filter(|_| state.client_hints),
        source_bytes,
        ops,
        merged_ops: resolved.ops,
        presets,
        canonical,
        cache_group: resolved.groups.into_iter().next(),
        image_path,
    })
}

/// The last stage of [`explain_handle`]: the report as the
/// client asked for it, JSON for a JSON `Accept` and the
/// HTML page otherwise, stored nowhere and served under a
/// policy that runs no script.
fn explain_response(report: &crate::explain::Report, accept: Option<&str>) -> Response {
    let (content_type, body) = if crate::explain::wants_json(accept) {
        (
            "application/json",
            serde_json::to_string(&report.to_json()).unwrap_or_default(),
        )
    } else {
        ("text/html; charset=utf-8", report.to_html())
    };
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-store"),
            (header::CONTENT_SECURITY_POLICY, "default-src 'none'"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        body,
    )
        .into_response()
}

/// A probe failure on the explain route: the image path's
/// statuses, with generic bodies throughout. The detailed
/// cause goes to the server log only: a policy refusal can
/// name a resolved address or a bucket, which are the
/// operator's facts, not the caller's (issue #196,
/// requirement 3).
fn explain_load_error(e: LoadError, accept: Option<&str>) -> Response {
    if let Some(busy) = load_busy_response(&e, accept) {
        return busy;
    }
    stderr_line!("explain load error: {e}");
    let (status, code, message) = load_error_parts(e);
    error_response(status, code, message, accept)
}
