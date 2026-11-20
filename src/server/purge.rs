//! Purges (issue #741): the per-URL and group purge handlers,
//! the purge expiry check, the tombstones and their refresh,
//! the purge of each tier (memory, disk, and shared with its
//! walks), the purge's log line, and the purge of a replaced
//! source that uploads run.
//!
//! server.rs re-exports what the router, main and the tests
//! use.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

use super::response::{ErrorParts, error_response};
use super::tiers::DISK_WRITE_WAIT;
use super::{AppState, Background};
use crate::observe::RequestLog;
use crate::url::{parse_operations_with_expiry, parse_signature};

/// How many shared-tier objects one purge request scans
/// inline (issue #368): the shared walk reads one entry
/// header per object (a bounded number at a time, issue
/// #462), so its inline cost must not grow with the whole
/// bucket. Nor does its listing (issue #709): the request
/// lists one page, of no more names than this. A walk that
/// outgrows the budget finishes in a background task,
/// waiting for one of the
/// [`crate::s3store::PURGE_WALKS_AT_ONCE`] turns with the
/// listing's continuation token, and logs its own count on
/// completion. Correctness does not wait
/// for it: the generation tombstone keeps writes begun
/// before the purge out (issue #350), and the purge's
/// stamped tombstone, here and in the bucket, makes every
/// copy written before the purge began a miss on every
/// replica until the walk has removed it (issues #400,
/// #461).
pub(in crate::server) const SHARED_PURGE_INLINE_SCAN: usize = 128;

/// How often a replica re-reads the purge tombstones in the
/// shared bucket (issue #461): the longest another replica's
/// purge can take to reach this one's lookups.
const TOMBSTONE_REFRESH: std::time::Duration = std::time::Duration::from_secs(5);

/// Start a purge of `target`: record it in this process's
/// guard and, with a shared tier, as a tombstone in the
/// bucket (issue #461), before any tier is touched. The
/// bucket record is what keeps the purge effective on
/// other replicas, and on this one after a restart, while
/// its walk runs. Returns the running purge and the
/// tombstone's object name, `None` when there is no shared
/// tier or the write failed (counted and logged by the
/// store; the purge then holds on this process only).
pub(in crate::server) async fn begin_purge(
    state: &AppState,
    target: crate::cache::PurgeTarget,
) -> (crate::cache::ActivePurge, Option<String>) {
    let active = state.purge_guard.begin(target);
    let tombstone = match &state.shared_store {
        Some(shared) => {
            shared
                .put_tombstone(active.tombstone(), active.sequence())
                .await
        }
        None => None,
    };
    (active, tombstone)
}

/// The shared tier's half of a purge (issues #368, #446,
/// #462): remove the covered copies written before the
/// purge began. A per-URL purge first deletes the current
/// epoch's variants by name. Then, when no identical purge
/// is walking the bucket in this process, this one leads a
/// walk: it scans at most [`SHARED_PURGE_INLINE_SCAN`]
/// objects inline and hands the remainder, if any, to a
/// background task that waits for a walk turn, tracked on
/// `background` (issue #463). When one is
/// walking, this purge joins that walk instead (see
/// [`crate::s3store::S3Store::lead_or_join`]), which
/// re-lists the bucket for it once its current pass ends.
/// A purge stays running until the walk that covers it
/// ends, then its tombstone is marked ended if the walk
/// reached every object (see [`end_purge`]). Returns what
/// the inline part removed.
pub(in crate::server) async fn purge_shared_tier(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    log: &RequestLog,
    begun: BegunPurge,
    epoch_base: Option<String>,
    event: &'static str,
    log_target: &str,
) -> usize {
    let BegunPurge {
        active,
        tombstone,
        disk,
    } = begun;
    let target = active.tombstone().target.clone();
    let cutoff = active.tombstone().cutoff;
    // The walk lists these variants and reads them again, so
    // a read here that fails is one the walk repairs.
    let mut removed = match &epoch_base {
        Some(epoch_base) => {
            shared
                .purge_epoch_variants(epoch_base, &target, cutoff)
                .await
                .removed
        }
        None => 0,
    };
    let purge = crate::s3store::SharedPurge {
        active,
        tombstone,
        disk,
        event,
        log_target: log_target.to_string(),
    };
    let Some((mut lease, purge)) = shared.lead_or_join(purge) else {
        return removed;
    };
    let (inline, pending) = shared
        .purge_bounded(&target, None, cutoff, SHARED_PURGE_INLINE_SCAN)
        .await;
    removed += inline.removed;
    let shared = shared.clone();
    let tasks = background.clone();
    let log = log.clone();
    match pending {
        Some(pending) => {
            background.spawn_walk(async move {
                let rest = {
                    let _turn = shared.walk_turn().await;
                    shared.finish_purge(pending).await
                };
                let (event, log_target) = (purge.event, purge.log_target.clone());
                end_purge_apart(&shared, &tasks, purge, inline.complete && rest.complete).await;
                purge_log(&log, event, &log_target, Some(rest.removed)).await;
                let joined = lease.take_joined();
                walk_for_joined(&shared, &tasks, &log, lease, joined).await;
            });
        }
        None => {
            end_purge_apart(&shared, background, purge, inline.complete).await;
            let joined = lease.take_joined();
            if !joined.is_empty() {
                background.spawn_walk(async move {
                    walk_for_joined(&shared, &tasks, &log, lease, joined).await;
                });
            }
        }
    }
    removed
}

/// Walk the bucket for the purges that joined a running
/// walk of their target (issue #462), one pass for all of
/// them, then again for any that joined during that pass,
/// until none is waiting. Each pass waits for a walk turn,
/// lists the bucket afresh, and judges copies by the
/// latest joined purge's start: the purges come from one
/// process, so its cutoff covers every copy the earlier
/// ones do. Each purge the pass served logs the pass's
/// count under its own target.
async fn walk_for_joined(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    log: &RequestLog,
    mut lease: crate::s3store::WalkLease,
    mut joined: Vec<crate::s3store::SharedPurge>,
) {
    while let Some(latest) = joined
        .iter()
        .max_by_key(|purge| purge.active.tombstone().cutoff.start_ms)
    {
        let target = latest.active.tombstone().target.clone();
        let cutoff = latest.active.tombstone().cutoff;
        let swept = {
            let _turn = shared.walk_turn().await;
            shared.relist(&target, cutoff).await
        };
        for purge in joined {
            let (event, log_target) = (purge.event, purge.log_target.clone());
            end_purge_apart(shared, background, purge, swept.complete).await;
            purge_log(log, event, &log_target, Some(swept.removed)).await;
        }
        joined = lease.take_joined();
    }
}

/// End a purge whose shared walk is done (issue #461):
/// rewrite its bucket tombstone with the walk's end, from
/// which every replica counts the grace period down and
/// then collects it, and end the purge in this process,
/// where its tombstone likewise outlives it for the grace.
/// A tombstone that is never marked (the process stopped
/// mid-walk) stays in force until a later purge of the same
/// target finishes its walk. So does the tombstone of a
/// walk that was not `complete` (issue #462): the store
/// failed to list the bucket or to answer for an object, so
/// a copy the purge covers may still be there, and it must
/// stay a miss after the grace. The tombstone is marked
/// ended only once the disk files of every purge of its
/// target in this process are deleted (issue #572): a
/// restart before then indexes them again, and the unended
/// tombstone keeps them misses. The purge waits for its own
/// while they make progress; when they stop, or one fails,
/// it ends here, and its tombstone is marked by the
/// tombstone refresh after they are gone.
async fn end_purge(
    shared: &crate::s3store::S3Store,
    purge: crate::s3store::SharedPurge,
    complete: bool,
) {
    if let Some(disk) = &purge.disk {
        disk.wait(PURGE_END_IDLE, PURGE_END_CAP).await;
    }
    if complete && let Some(object) = &purge.tombstone {
        shared.end_tombstone(object, purge.active.tombstone()).await;
    }
    drop(purge);
}

/// [`end_purge`], on a task of its own when the purge still
/// waits for disk deletions (issue #572): the walk lease, and
/// the purges that joined it, do not wait on the disk. A
/// purge with no tombstone to mark ended waits for nothing;
/// its deletions hold the target's tombstones from the moment
/// they were queued (see [`purge_disk_tier`]).
async fn end_purge_apart(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    purge: crate::s3store::SharedPurge,
    complete: bool,
) {
    if !complete || purge.tombstone.is_none() {
        return;
    }
    if purge.disk.as_ref().is_some_and(|disk| disk.pending()) {
        let shared = shared.clone();
        background.spawn_walk(async move { end_purge(&shared, purge, complete).await });
    } else {
        end_purge(shared, purge, complete).await;
    }
}

/// A purge that has begun, as its shared-tier half takes it:
/// the running purge, its bucket tombstone, and the disk
/// deletions its tombstone waits for (issue #572).
pub(in crate::server) struct BegunPurge {
    pub(in crate::server) active: crate::cache::ActivePurge,
    pub(in crate::server) tombstone: Option<String>,
    pub(in crate::server) disk: Option<crate::storage::DeletionWait>,
}

/// The disk tier's half of a purge (issue #572): remove the
/// records `target` covers from the index at once, so they
/// stop serving and the count is exact, and queue their
/// files' deletion. No disk slot is waited for. The deletions
/// hold the target's bucket tombstones at once: a per-URL
/// purge awaits the bucket before its walk, and a later purge
/// of the URL that ends meanwhile, with no deletions of its
/// own, must not mark its tombstone ended while these files
/// are there. Returns the count and the deletions queued.
pub(in crate::server) fn purge_disk_tier(
    state: &AppState,
    target: &crate::cache::PurgeTarget,
) -> (usize, Option<crate::storage::DeletionWait>) {
    let Some(store) = &state.store else {
        return (0, None);
    };
    let (removed, disk) = store.purge_waiting(target);
    store.kick();
    if let Some(shared) = &state.shared_store {
        shared.hold_for_deletions(target, &disk);
    }
    (removed, Some(disk))
}

/// Wait for a purge's disk deletions (issue #572) while they
/// make progress: on a healthy disk with no deletions queued
/// before them, the files are gone when the purge answers, as
/// before. Once none finishes for [`DISK_WRITE_WAIT`], or after
/// [`PURGE_DELETE_CAP`] in all, the purge answers and the
/// deletions go on in the background. That counts as a stall
/// only on a stuck disk, not on one busy with the deletions
/// queued before these.
pub(in crate::server) async fn wait_for_disk_purge(disk: Option<crate::storage::DeletionWait>) {
    let Some(disk) = disk else {
        return;
    };
    let waited = disk.wait(DISK_WRITE_WAIT, PURGE_DELETE_CAP).await;
    if waited == crate::storage::Waited::Stalled {
        disk.store.note_stall(crate::storage::Stall::PurgeDelete);
    }
}

/// Load the shared bucket's purge tombstones before serving
/// (issue #461), then keep them fresh every
/// `TOMBSTONE_REFRESH` on one background task. Without a
/// shared tier there is nothing to read. Each refresh, the
/// boot one included, ends within
/// [`crate::s3store::TOMBSTONE_REFRESH_DEADLINE`] (issue
/// #549), so a slow or hanging store delays the first
/// request by that much at most. A failed or unfinished
/// boot read is counted and logged, and the refresher
/// retries: the tier stays best-effort, like every other
/// shared read. Serving does not start first: lookups
/// consult the tombstones, so a request served before them
/// could hand out a copy a purge removed. Each refresh after
/// boot first marks ended the tombstones that waited for disk
/// deletions now done (issue #572).
pub async fn start_purge_tombstones(state: &AppState) {
    let Some(shared) = state.shared_store.clone() else {
        return;
    };
    shared.refresh_tombstones(&state.purge_guard).await;
    let guard = state.purge_guard.clone();
    // One task per process, started at boot and never
    // ending, so it stays off the shutdown drain (issue
    // #463), which waits only for work that finishes; it
    // stops with the process.
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(TOMBSTONE_REFRESH);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            shared.end_held_purges().await;
            shared.refresh_tombstones(&guard).await;
        }
    });
}

/// One structured line per purge request (audit PL-8): the
/// destructive, secret-gated endpoint must be auditable. The
/// target is hashed; a raw target can carry signed URLs.
///
/// The line is on standard output when this returns, so
/// before the purge is answered (see [`RequestLog::event`]).
async fn purge_log(log: &RequestLog, event: &str, target: &str, removed: Option<usize>) {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(target.as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let line = match removed {
        Some(n) => format!("{{\"event\":\"{event}\",\"target\":\"{hash}\",\"removed\":{n}}}"),
        None => format!("{{\"event\":\"{event}\",\"target\":\"{hash}\",\"rejected\":true}}"),
    };
    log.event(&line).await;
}

/// The current unix second, as the purge expiry check
/// reads it.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Why a purge whose signature has verified may not run, by
/// its expiry, at the unix second `now`. A purge URL is
/// valid strictly before its `exp:` second, the comparison
/// image and upload URLs use: from that second on it is
/// refused as expired. One with no expiry runs, unless
/// `require_expiry` (`CRABCROP_PURGE_REQUIRE_EXPIRY`) says
/// the server takes only expiring purge URLs.
pub(in crate::server) fn purge_expiry_refusal(
    expiry: Option<u64>,
    now: u64,
    require_expiry: bool,
) -> Option<ErrorParts> {
    match expiry {
        Some(exp) if now >= exp => Some((
            StatusCode::FORBIDDEN,
            "signature_expired",
            "signature expired".to_string(),
        )),
        Some(_) => None,
        None if require_expiry => Some((
            StatusCode::FORBIDDEN,
            "expiry_required",
            "expiry required: this server accepts only purge URLs that carry an exp: segment"
                .to_string(),
        )),
        None => None,
    }
}

/// `POST /purge/<signature>/<path>`: invalidate every stored
/// variant of one result. The removal spans this process's
/// tiers (its memory, its disk store, and the shared store);
/// it does not reach another process's memory, so a replica
/// can keep serving its own copy until it evicts or restarts
/// (issue #351). The signature is an HMAC over the
/// path under a purge subkey derived from the secret (see
/// [`crate::url::purge_key`]), so only the secret holder can
/// purge and an image signature does not verify here (no
/// signer signs the subkey's label); a bad signature is a
/// 403 and purges nothing. The expiring form,
/// `POST /purge/<signature>/exp:<unix-seconds>/<path>`, signs
/// the `exp:` segment along with the path, and is a 403
/// from its expiry second on (see
/// [`crate::url::split_purge_remainder`]); with
/// `CRABCROP_PURGE_REQUIRE_EXPIRY`, a URL without an expiry
/// is a 403 as well. Idempotent:
/// purging an absent entry answers 200 with a count of 0.
///
/// The purge runs in stages: [`admit_purge`] checks the
/// signature and the expiry, [`purge_key_form`] names the
/// entries the target is stored under, and [`fan_out_purge`]
/// removes them from every tier, in a task of its own.
pub(in crate::server) async fn purge_handle(
    State(state): State<Arc<AppState>>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    // A purge leaves a log line: like an image request, it
    // waits while the request log's queue is full.
    state.request_log.room().await;
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(tail) = uri.path().strip_prefix("/purge/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /purge/<signature>/<target>".to_string(),
            accept.as_deref(),
        );
    };
    let full = format!("/{tail}");
    let target = match admit_purge(&state, &full) {
        Ok(target) => target,
        Err(PurgeRefusal::Malformed((status, code, message))) => {
            return error_response(status, code, message, accept.as_deref());
        }
        Err(PurgeRefusal::Rejected(logged, (status, code, message))) => {
            state.metrics.record_purge();
            purge_log(&state.request_log, "purge", logged, None).await;
            return error_response(status, code, message, accept.as_deref());
        }
    };
    let (target, key_form) = purge_key_form(&state, target);
    // The fan-out runs in a spawned task once the
    // signature has verified (issue #347): a purge that
    // has started completes and is logged even if the
    // caller disconnects mid-purge; dropping the
    // JoinHandle detaches the task instead of cancelling
    // it. The caller waits for the count, so the response
    // still carries it. The task is tracked (issue #463),
    // so a shutdown waits for a purge whose caller left.
    let task = state
        .background
        .spawn(fan_out_purge(state.clone(), key_form, target));
    let removed = task.await.unwrap_or(0);
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// Why [`admit_purge`] turned a purge request away.
enum PurgeRefusal<'a> {
    /// The URL does not parse: it is answered and nothing
    /// else is done.
    Malformed(ErrorParts),
    /// The signature or the expiry refused it: the purge is
    /// counted and logged as rejected under the target named
    /// here, then answered.
    Rejected(&'a str, ErrorParts),
}

/// The first stage of [`purge_handle`]: parse the signature
/// segment of `full` (the path after `/purge`), verify it
/// over the remainder as presented, split the purge's own
/// expiry off the front, and check it. Returns the target
/// that follows the expiry.
fn admit_purge<'a>(state: &AppState, full: &'a str) -> Result<&'a str, PurgeRefusal<'a>> {
    let (signature, key_id, presented) = match parse_signature(full) {
        Ok(v) => v,
        Err(e) => {
            return Err(PurgeRefusal::Malformed((
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
            )));
        }
    };
    // Verification runs over the remainder exactly as
    // presented: for the expiring form the `exp:` segment is
    // part of the signed string, so a tampered or stripped
    // expiry already fails here as a bad signature.
    if !state
        .keyring
        .verify_purge(key_id.as_deref(), &signature, presented)
    {
        return Err(PurgeRefusal::Rejected(
            presented,
            (
                StatusCode::FORBIDDEN,
                "bad_signature",
                "bad signature".to_string(),
            ),
        ));
    }
    // A leading `exp:` segment is the purge's own expiry,
    // split off the verified remainder as the upload
    // endpoint splits its own, and enforced before anything
    // is touched. What follows it is the target.
    let (expiry, target) = match crate::url::split_purge_remainder(presented) {
        Ok(v) => v,
        Err(e) => {
            return Err(PurgeRefusal::Malformed((
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
            )));
        }
    };
    if let Some(refusal) = purge_expiry_refusal(expiry, unix_now(), state.purge_require_expiry) {
        return Err(PurgeRefusal::Rejected(target, refusal));
    }
    Ok(target)
}

/// The second stage of [`purge_handle`]: the target the
/// purge logs, and the key form of the entries it names,
/// for a verified target. The target unseals, then expands
/// its presets and its source root, as the image path would
/// expand it; the key form is the canonical form of its
/// operations merged with the defaults.
fn purge_key_form(state: &AppState, target: &str) -> (String, String) {
    // A sealed target (issue #255) unseals here, the first
    // step of the chain, now that the purge signature has
    // verified over the target as presented and the purge's
    // own expiry is off its front: the steps below continue
    // with the decrypted remainder, so a purge by the sealed
    // spelling evicts the entry its plain spelling serves,
    // and the reverse. An `exp:` inside the payload is the
    // image's, which the key form drops. The decrypted
    // remainder replaces the target wholesale (anything
    // after the token segment was the cosmetic friendly
    // name). A token that does not open under this keyring
    // falls back to the presented target, which matches
    // nothing: the same quiet no-op an unverifiable spelling
    // gets everywhere else in this handler.
    let unsealed;
    let target = {
        let presented_segment = target.split('/').next().unwrap_or("");
        match crate::url::split_sealed_segment(presented_segment) {
            Some((kid, token)) if presented_segment.len() <= crabcrop_url::MAX_SEALED_SEGMENT => {
                match state.keyring.open_sealed(kid, token) {
                    Some(text) => {
                        unsealed = text;
                        unsealed.as_str()
                    }
                    None => target,
                }
            }
            _ => target,
        }
    };
    // The signature covers the target as presented; the
    // entries, though, are keyed by the canonical form of
    // the EXPANDED remainder (Q1, Q5). Expand presets the
    // way serve does, or a purge of the preset URL an
    // operator actually holds silently removes 0 entries.
    // A target that does not expand (a directly expanded
    // path, an unknown preset) falls back to itself.
    let expanded;
    let target = match crate::url::expand_presets(target, &state.presets, state.only_presets) {
        Ok(text) => {
            expanded = text;
            &expanded
        }
        Err(_) => target,
    };
    // Root sources expand the same way (issue #248): the
    // entry is keyed on the expansion, so a purge naming
    // the short spelling must reach it.
    let root_expanded;
    let target = match crate::url::expand_source_root_in_remainder(target, &state.source_roots) {
        Ok(text) => {
            root_expanded = text;
            &root_expanded
        }
        Err(_) => target,
    };
    // Entries are keyed post-merge (issue #145): with
    // defaults configured, the purge target's key form is
    // the canonical form of the merged operations, or the
    // purge silently removes nothing. The form is bounded
    // the way `cache_key_for` bounds it, so an over-long
    // URL's purge names its digest-form key.
    let key_form = match parse_operations_with_expiry(target) {
        Ok((ops, source, _, _)) => {
            let merged = crate::url::merge_default_ops(&state.default_ops, ops);
            crate::cache::bounded_key_base(crate::url::canonical_form(&merged, source), source)
        }
        Err(_) => crate::cache::bounded_key_base(target.to_string(), ""),
    };
    (target.to_string(), key_form)
}

/// The last stage of [`purge_handle`], the task it spawns:
/// remove the entries `key_form` names from every tier of
/// this process, then count the purge and log it under
/// `target`. Returns how many entries it removed.
async fn fan_out_purge(task_state: Arc<AppState>, key_form: String, task_target: String) -> usize {
    // The tombstones land before any tier is touched
    // (issues #350, #461): a write begun before this
    // purge can then see it whenever it tries to land,
    // including after this handler has finished, and
    // every replica treats the copies it covers as
    // purged until the shared walk ends (issue #400).
    let (active, tombstone) = begin_purge(
        &task_state,
        crate::cache::PurgeTarget::Key(key_form.clone()),
    )
    .await;
    let mut removed = task_state.cache.purge_matching(&key_form);
    let (on_disk, disk) = purge_disk_tier(
        &task_state,
        &crate::cache::PurgeTarget::Key(key_form.clone()),
    );
    removed += on_disk;
    if let Some(shared) = &task_state.shared_store {
        // The shared tier deletes the current epoch's
        // enumerable variants directly by object name
        // (audit R2-24); the fingerprint-free form drives
        // its matching walk, which also reaches older
        // epochs and hint-variant spellings.
        let epoch_form = format!("{key_form}#cfg{}", task_state.config_fingerprint);
        removed += purge_shared_tier(
            shared,
            &task_state.background,
            &task_state.request_log,
            BegunPurge {
                active,
                tombstone,
                disk: disk.clone(),
            },
            Some(epoch_form),
            "purge",
            &task_target,
        )
        .await;
    }
    wait_for_disk_purge(disk).await;
    task_state.metrics.record_purge();
    purge_log(
        &task_state.request_log,
        "purge",
        &task_target,
        Some(removed),
    )
    .await;
    removed
}

/// `POST /purge-group/<signature>/<group>`: invalidate every
/// stored entry in one cache group (RFC 9875), across this
/// process's tiers (its memory, its disk store, and the
/// shared store); another process's memory is not reached,
/// as in the per-URL purge (issue #351). The signature is
/// verified under the purge-group
/// subkey, its own domain beside the purge subkey, over the
/// group name, so a per-URL purge signature does not verify
/// here, and neither does an image signature (no signer
/// signs the subkey's label). A bad signature is a 403 and
/// removes nothing; an unknown group answers 200 with a
/// count of 0. The expiring form,
/// `POST /purge-group/<signature>/exp:<unix-seconds>/<group>`,
/// signs the `exp:` segment along with the group name, and
/// is a 403 from its expiry second on; the setting that
/// refuses a URL without one applies here too.
pub(in crate::server) async fn purge_group_handle(
    State(state): State<Arc<AppState>>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    // As in `purge_handle`.
    state.request_log.room().await;
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(tail) = uri.path().strip_prefix("/purge-group/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /purge-group/<signature>/<group>".to_string(),
            accept.as_deref(),
        );
    };
    let full = format!("/{tail}");
    let (signature, key_id, presented) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
                accept.as_deref(),
            );
        }
    };
    // The group's shape is checked before its signature, as
    // it always was, so the purge's own expiry comes off the
    // front first (see `crate::url::split_purge_remainder`):
    // what follows it is the group name.
    let (expiry, group) = match crate::url::split_purge_remainder(presented) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
                accept.as_deref(),
            );
        }
    };
    if group.is_empty() || group.contains('/') {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_purge_group",
            "invalid purge group name".to_string(),
            accept.as_deref(),
        );
    }
    let target = format!("purge-group/{group}");
    // The signature covers the remainder as presented, the
    // `exp:` segment included.
    if !state
        .keyring
        .verify_purge_group(key_id.as_deref(), &signature, presented)
    {
        state.metrics.record_purge();
        purge_log(&state.request_log, "purge-group", &target, None).await;
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    if let Some((status, code, message)) =
        purge_expiry_refusal(expiry, unix_now(), state.purge_require_expiry)
    {
        state.metrics.record_purge();
        purge_log(&state.request_log, "purge-group", &target, None).await;
        return error_response(status, code, message, accept.as_deref());
    }
    // The fan-out runs in a spawned task once the
    // signature has verified, as in the per-URL purge
    // (issue #347).
    let task_state = state.clone();
    let task_target = target.clone();
    let group = group.to_string();
    let task = state.background.spawn(async move {
        // Tombstones first, tiers second (issues #350,
        // #461), as in the per-URL purge.
        let (active, tombstone) =
            begin_purge(&task_state, crate::cache::PurgeTarget::Group(group.clone())).await;
        let mut removed = task_state.cache.purge_group(&group);
        let (on_disk, disk) = purge_disk_tier(
            &task_state,
            &crate::cache::PurgeTarget::Group(group.clone()),
        );
        removed += on_disk;
        if let Some(shared) = &task_state.shared_store {
            removed += purge_shared_tier(
                shared,
                &task_state.background,
                &task_state.request_log,
                BegunPurge {
                    active,
                    tombstone,
                    disk: disk.clone(),
                },
                None,
                "purge-group",
                &task_target,
            )
            .await;
        }
        wait_for_disk_purge(disk).await;
        task_state.metrics.record_purge();
        purge_log(
            &task_state.request_log,
            "purge-group",
            &task_target,
            Some(removed),
        )
        .await;
        removed
    });
    let removed = task.await.unwrap_or(0);
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// Remove every cached derivative of `source` from all
/// three of this process's tiers; returns the total
/// removed. The upload
/// overwrite path (audit R2-07) shares this with any
/// future source-replacement flow.
pub(in crate::server) async fn purge_source_everywhere(
    state: &Arc<AppState>,
    source: &str,
) -> usize {
    // The fan-out runs in a spawned task, as in the purge
    // handlers (issue #347): a purge that has started
    // completes and is logged even if the caller is
    // dropped mid-purge. The caller waits for the count.
    let task_state = state.clone();
    let source = source.to_string();
    let task = state.background.spawn(async move {
        // Tombstones first, tiers second (issues #350,
        // #461), as in the per-URL purge.
        let (active, tombstone) = begin_purge(
            &task_state,
            crate::cache::PurgeTarget::Source(source.clone()),
        )
        .await;
        let mut removed = task_state.cache.purge_source(&source);
        let (on_disk, disk) = purge_disk_tier(
            &task_state,
            &crate::cache::PurgeTarget::Source(source.clone()),
        );
        removed += on_disk;
        if let Some(shared) = &task_state.shared_store {
            removed += purge_shared_tier(
                shared,
                &task_state.background,
                &task_state.request_log,
                BegunPurge {
                    active,
                    tombstone,
                    disk: disk.clone(),
                },
                None,
                "purge-source",
                &source,
            )
            .await;
        }
        wait_for_disk_purge(disk).await;
        purge_log(
            &task_state.request_log,
            "purge-source",
            &source,
            Some(removed),
        )
        .await;
        removed
    });
    task.await.unwrap_or(0)
}

/// How long a purge waits for its disk deletions in all
/// while they make progress (issue #572). It stops sooner, at
/// [`DISK_WRITE_WAIT`], once none of them finishes for that
/// long.
const PURGE_DELETE_CAP: std::time::Duration = std::time::Duration::from_secs(30);

/// How long the end of a purge, in the background, waits for
/// one of its disk deletions to finish (issue #572). When none
/// does for that long, the purge ends, and its bucket
/// tombstone is marked ended by the tombstone refresh once
/// they are done.
const PURGE_END_IDLE: std::time::Duration = std::time::Duration::from_secs(30);

/// How long the end of a purge waits for its disk deletions
/// in all while they make progress.
const PURGE_END_CAP: std::time::Duration = std::time::Duration::from_secs(600);
