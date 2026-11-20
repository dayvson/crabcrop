use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

use super::AppState;
use super::response::{accept_of, error_response};

/// The health check: `200 ok`, unless the request log has
/// stalled (issue #707). Its queue has then been full for
/// [`crate::observe::STALLED_AFTER`], so every image
/// request, purge and upload since has waited for a
/// standard output nobody reads, and the replica serves
/// none of them: the check answers `503
/// request_log_stalled`, so a liveness probe restarts the
/// replica, as it did before the queue (issue #642), when
/// such a standard output held the whole server, this
/// check included. The requests keep waiting and no line is
/// dropped. The check reads two atomics and never waits on
/// the queue.
pub(in crate::server) async fn health(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let Some(full) = state.request_log.stalled() else {
        return "ok".into_response();
    };
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "request_log_stalled",
        format!(
            "request log stalled: its queue has been full for {}s, and requests wait for standard output to be read",
            full.as_secs()
        ),
        accept_of(&headers).as_deref(),
    )
}

/// Equality for a bearer token that leaks nothing about it
/// through timing, its length included (issue #454): both
/// sides are hashed first, so the comparison always runs
/// over two 32-byte digests, however long the presented
/// token is.
pub(in crate::server) fn token_eq(a: &[u8], b: &[u8]) -> bool {
    use sha2::Digest as _;
    let (a, b) = (sha2::Sha256::digest(a), sha2::Sha256::digest(b));
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

pub(in crate::server) async fn metrics(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    // The exposition is operational data (issue #270,
    // the PL-5 residual): it serves only to a bearer of
    // the configured token, and an unconfigured server
    // keeps it closed rather than public.
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(token) = &state.metrics_token else {
        return error_response(
            StatusCode::FORBIDDEN,
            "metrics_disabled",
            "metrics are disabled: set CRABCROP_METRICS_TOKEN to enable them".to_string(),
            accept.as_deref(),
        );
    };
    // The auth scheme is case-insensitive (RFC 9110), so
    // `bearer` and `BEARER` name it too.
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, credentials)| credentials);
    if !presented.is_some_and(|p| token_eq(p.as_bytes(), token.as_bytes())) {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "metrics_unauthorized",
            "metrics require a bearer token".to_string(),
            accept.as_deref(),
        );
    }
    // One buffer for the whole exposition (issue #643):
    // each section appends its lines to it.
    let mut out = String::with_capacity(4096);
    state.metrics.render_into(&mut out);
    lane_metrics(&state, &mut out);
    gate_metrics(&state, &mut out);
    ai_metrics(&mut out);
    request_log_metrics(&state, &mut out);
    source_buffer_metrics(&state, &mut out);
    source_lookup_metrics(&state, &mut out);
    tier_metrics(&state, &mut out);
    upload_metrics(&state, &mut out);
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], out).into_response()
}

/// The upload bucket's failed existence checks for `/metrics`
/// (issue #709), with a bucket destination: a HEAD the bucket
/// refused, failed or did not answer reads as no object, so a
/// bucket that stops answering them shows here, not in an
/// upload's answer. The upload bucket's store is not the shared
/// tier, and does not count in its series.
fn upload_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    let Some(failed) = state
        .upload
        .as_ref()
        .and_then(|upload| upload.exists_checks_failed())
    else {
        return;
    };
    let _ = write!(
        out,
        "# HELP crabcrop_upload_exists_checks_failed_total Upload existence checks the upload bucket refused, failed or did not answer; each read as no object.\n\
         # TYPE crabcrop_upload_exists_checks_failed_total counter\n\
         crabcrop_upload_exists_checks_failed_total {failed}\n"
    );
}

/// Per-lane occupancy for `/metrics` (issue #324): how
/// many permits of each processing lane are in use,
/// against the lane's capacity. Read live from the gates
/// at scrape time, so a lane saturated by long encodes
/// shows itself here while the stills lane stays free.
fn lane_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    out.push_str("# HELP crabcrop_lane_in_use Processing lane permits currently in use.\n");
    out.push_str("# TYPE crabcrop_lane_in_use gauge\n");
    let _ = writeln!(
        out,
        "crabcrop_lane_in_use{{lane=\"process\"}} {}",
        state.process_gate.in_use()
    );
    let _ = writeln!(
        out,
        "crabcrop_lane_in_use{{lane=\"slow\"}} {}",
        state.slow_gate.in_use()
    );
    out.push_str("# HELP crabcrop_lane_capacity Processing lane permit capacity.\n");
    out.push_str("# TYPE crabcrop_lane_capacity gauge\n");
    let _ = writeln!(
        out,
        "crabcrop_lane_capacity{{lane=\"process\"}} {}",
        state.process_gate.capacity()
    );
    let _ = writeln!(
        out,
        "crabcrop_lane_capacity{{lane=\"slow\"}} {}",
        state.slow_gate.capacity()
    );
}

/// Gate occupancy for `/metrics` (issue #707): how many
/// permits of the AI, video, animation and `/explain` probe
/// gates are in use, against each gate's capacity, read live
/// at scrape time as the lanes are. A gate at capacity while
/// requests shed with `server_busy` names the work they
/// waited for.
fn gate_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    let gates = [
        ("ai", &state.ai_gate),
        ("video", &state.video_gate),
        ("animation", &state.anim_gate),
        ("probe", &state.probe_gate),
    ];
    out.push_str("# HELP crabcrop_gate_in_use Gate permits currently in use.\n");
    out.push_str("# TYPE crabcrop_gate_in_use gauge\n");
    for (name, gate) in gates {
        let _ = writeln!(
            out,
            "crabcrop_gate_in_use{{gate=\"{name}\"}} {}",
            gate.in_use()
        );
    }
    out.push_str("# HELP crabcrop_gate_capacity Gate permit capacity.\n");
    out.push_str("# TYPE crabcrop_gate_capacity gauge\n");
    for (name, gate) in gates {
        let _ = writeln!(
            out,
            "crabcrop_gate_capacity{{gate=\"{name}\"}} {}",
            gate.capacity()
        );
    }
}

/// AI model lock recoveries for `/metrics` (issue #736): the
/// locks on the models' sessions and memos that a request's
/// panic poisoned and a later request took back (issue #715),
/// counted since boot. Each is also a line on stderr; a count
/// that keeps rising is a model that panics on what it is
/// given.
fn ai_metrics(out: &mut String) {
    use std::fmt::Write as _;
    let _ = write!(
        out,
        "# HELP crabcrop_ai_lock_recoveries_total AI model locks taken back after a request panicked while it held one.\n\
         # TYPE crabcrop_ai_lock_recoveries_total counter\n\
         crabcrop_ai_lock_recoveries_total {}\n",
        crate::ai::lock_recoveries()
    );
}

/// The request log's queue for `/metrics` (issue #707): the
/// bytes of lines it holds against what it holds before
/// requests wait, the requests waiting for room, and the
/// seconds it has spent full. A queue that stays full is a
/// standard output nobody reads, and the health check fails
/// on it after `STALLED_AFTER`; bytes held that never fall
/// to 0 between scrapes are one whose reader has stopped,
/// even before the queue fills. With `CRABCROP_LOG_SYNC`
/// there is no queue, and none of these series.
fn request_log_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    let Some(queue) = state.request_log.queue_state() else {
        return;
    };
    let full_ms = queue.full_time.as_millis();
    let _ = write!(
        out,
        "# HELP crabcrop_request_log_queue_bytes Bytes of request log lines held for standard output, queued or being written.\n\
         # TYPE crabcrop_request_log_queue_bytes gauge\n\
         crabcrop_request_log_queue_bytes {}\n\
         # HELP crabcrop_request_log_queue_capacity_bytes The bytes the request log holds before requests wait for room.\n\
         # TYPE crabcrop_request_log_queue_capacity_bytes gauge\n\
         crabcrop_request_log_queue_capacity_bytes {}\n\
         # HELP crabcrop_request_log_requests_waiting Requests waiting for room in the request log.\n\
         # TYPE crabcrop_request_log_requests_waiting gauge\n\
         crabcrop_request_log_requests_waiting {}\n\
         # HELP crabcrop_request_log_queue_full_seconds_total Seconds the request log's queue has spent full, when requests that log wait for room.\n\
         # TYPE crabcrop_request_log_queue_full_seconds_total counter\n\
         crabcrop_request_log_queue_full_seconds_total {}.{:03}\n",
        queue.held_bytes,
        queue.capacity,
        queue.waiting,
        full_ms / 1000,
        full_ms % 1000
    );
}

/// Source buffer budget occupancy for `/metrics` (issue
/// #403): the source and watermark bytes cache misses hold
/// right now, against the configured budget, read live at
/// scrape time. In use sitting at capacity while requests
/// shed with `server_busy` is the signal to raise
/// `CRABCROP_SOURCE_BUFFER_MAX_MB`.
fn source_buffer_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    let budget = state.loader.source_budget();
    let _ = write!(
        out,
        "# HELP crabcrop_source_buffer_bytes Source bytes cache misses hold against the source buffer budget.\n\
         # TYPE crabcrop_source_buffer_bytes gauge\n\
         crabcrop_source_buffer_bytes {}\n\
         # HELP crabcrop_source_buffer_capacity_bytes The source buffer budget.\n\
         # TYPE crabcrop_source_buffer_capacity_bytes gauge\n\
         crabcrop_source_buffer_capacity_bytes {}\n",
        budget.in_use(),
        budget.capacity()
    );
}

/// Source host lookup slot occupancy for `/metrics`: the
/// lookups running right now against the slots (see
/// [`crate::loader::LOOKUP_SLOTS`]), read live at scrape time.
/// In flight sitting at capacity while requests shed with
/// `server_busy` means the resolver is not answering for the
/// host names requests carry.
fn source_lookup_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    let (in_flight, capacity) = state.loader.lookup_slots();
    let _ = write!(
        out,
        "# HELP crabcrop_source_lookups_in_flight Source host lookups running now.\n\
         # TYPE crabcrop_source_lookups_in_flight gauge\n\
         crabcrop_source_lookups_in_flight {in_flight}\n\
         # HELP crabcrop_source_lookups_capacity Source host lookups that can run at once.\n\
         # TYPE crabcrop_source_lookups_capacity gauge\n\
         crabcrop_source_lookups_capacity {capacity}\n"
    );
}

/// Cache tier failures for `/metrics` (issue #367): the
/// stores count their own failures as they log them; the
/// exposition reads the counters live at scrape time,
/// one series per configured tier. With a shared tier, its
/// write-through slots and skips follow (issue #463), the
/// group write-backs waiting for a slot and skipped, and the
/// requests the bucket refused (issue #709), then what its
/// last complete listing of the purge tombstones found
/// unread, and when that was (issue #577).
fn tier_metrics(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    out.push_str("# HELP crabcrop_cache_tier_errors_total Cache tier operations that failed.\n");
    out.push_str("# TYPE crabcrop_cache_tier_errors_total counter\n");
    if let Some(store) = &state.store {
        let _ = writeln!(
            out,
            "crabcrop_cache_tier_errors_total{{tier=\"disk\"}} {}",
            store.errors()
        );
    }
    if let Some(store) = &state.shared_store {
        let _ = writeln!(
            out,
            "crabcrop_cache_tier_errors_total{{tier=\"shared\"}} {}",
            store.errors()
        );
    }
    // The disk tier's slots in use and its background work
    // (issue #572): queue lengths, never an index scan.
    if let Some(store) = &state.store {
        let (behind, deletions) = store.queued();
        let _ = write!(
            out,
            "# HELP crabcrop_cache_disk_ops_in_flight Disk tier operations holding one of its 32 slots.\n\
             # TYPE crabcrop_cache_disk_ops_in_flight gauge\n\
             crabcrop_cache_disk_ops_in_flight {}\n\
             # HELP crabcrop_cache_disk_groups_behind Disk entries whose file has not caught up with the cache groups they were served under.\n\
             # TYPE crabcrop_cache_disk_groups_behind gauge\n\
             crabcrop_cache_disk_groups_behind {behind}\n\
             # HELP crabcrop_cache_disk_deletions_queued Purged or evicted disk entries whose file is not deleted yet.\n\
             # TYPE crabcrop_cache_disk_deletions_queued gauge\n\
             crabcrop_cache_disk_deletions_queued {deletions}\n\
             # HELP crabcrop_cache_disk_unlinks_failed_total Disk entry files an inline delete or the boot scan failed to remove, left on disk out of the index.\n\
             # TYPE crabcrop_cache_disk_unlinks_failed_total counter\n\
             crabcrop_cache_disk_unlinks_failed_total {}\n",
            store.ops_in_flight(),
            store.unlinks_failed()
        );
    }
    if let Some(store) = &state.shared_store {
        // The write bound (issue #463): in flight against
        // its slots, and the render write-throughs skipped
        // because every slot was taken.
        let background = &state.background;
        let _ = write!(
            out,
            "# HELP crabcrop_cache_shared_puts_in_flight Shared tier writes in flight.\n\
             # TYPE crabcrop_cache_shared_puts_in_flight gauge\n\
             crabcrop_cache_shared_puts_in_flight {}\n\
             # HELP crabcrop_cache_shared_puts_capacity Shared tier writes allowed in flight.\n\
             # TYPE crabcrop_cache_shared_puts_capacity gauge\n\
             crabcrop_cache_shared_puts_capacity {}\n\
             # HELP crabcrop_cache_shared_puts_skipped_total Shared tier render write-throughs skipped with every slot in flight.\n\
             # TYPE crabcrop_cache_shared_puts_skipped_total counter\n\
             crabcrop_cache_shared_puts_skipped_total {}\n\
             # HELP crabcrop_cache_shared_group_writes_waiting Shared tier group write-backs waiting for a write slot.\n\
             # TYPE crabcrop_cache_shared_group_writes_waiting gauge\n\
             crabcrop_cache_shared_group_writes_waiting {}\n\
             # HELP crabcrop_cache_shared_group_writes_skipped_total Shared tier group write-backs skipped with as many waiting for a slot as may.\n\
             # TYPE crabcrop_cache_shared_group_writes_skipped_total counter\n\
             crabcrop_cache_shared_group_writes_skipped_total {}\n",
            background.puts_in_flight(),
            background.put_capacity(),
            background.puts_skipped(),
            background.group_writes_waiting(),
            background.group_writes_skipped()
        );
        // The requests the bucket refused (issue #709): a
        // lookup answered with an error that is neither a 404
        // nor a store failure, and a purge read or delete
        // refused.
        out.push_str(
            "# HELP crabcrop_cache_shared_refused_total Shared tier requests the bucket refused, by request.\n\
             # TYPE crabcrop_cache_shared_refused_total counter\n",
        );
        for kind in crate::s3store::Refused::ALL {
            let _ = writeln!(
                out,
                "crabcrop_cache_shared_refused_total{{request=\"{}\"}} {}",
                kind.label(),
                store.refused(kind)
            );
        }
        // The tombstone gap (issue #577): the unread count
        // stays as of the last complete listing, so a listing
        // time that stops moving is the sign of a failing
        // listing.
        let listed_ms = store.tombstones_listed_ms();
        let _ = write!(
            out,
            "# HELP crabcrop_cache_shared_tombstones_unread Purge tombstones the last complete listing named that this replica has never read; their purges do not apply here until read.\n\
             # TYPE crabcrop_cache_shared_tombstones_unread gauge\n\
             crabcrop_cache_shared_tombstones_unread {}\n\
             # HELP crabcrop_cache_shared_tombstones_listed_timestamp_seconds When this replica last listed the purge tombstones completely, in Unix seconds; 0 before its first complete listing.\n\
             # TYPE crabcrop_cache_shared_tombstones_listed_timestamp_seconds gauge\n\
             crabcrop_cache_shared_tombstones_listed_timestamp_seconds {}.{:03}\n",
            store.tombstones_unread(),
            listed_ms / 1000,
            listed_ms % 1000
        );
    }
}
