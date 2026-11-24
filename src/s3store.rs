//! Shared result storage (issue #142): the multi-replica tier
//! consulted after the local memory and disk caches, backed by
//! one S3 bucket (or an S3-compatible endpoint).
//!
//! Entries are the disk store's serialized form (the CRABDISK
//! bytes: key and cache groups embedded, MACed under a key
//! derived from the server secret), one object per entry,
//! named by the SHA-256 of the cache key under a configurable
//! prefix. Purges match on each entry's stored key and
//! groups, read from the entry header alone (a ranged read,
//! never the whole object; audit R2-24), and per-URL purges
//! additionally reach the target's enumerable key variants
//! directly by their deterministic object names. A purge
//! removes only the copies written before it began (issue
//! #446, by the write stamp in each header), and deletes
//! each one conditionally on the ETag it read, so a fresh
//! render that replaces the object meanwhile survives.
//!
//! Purges also leave tombstones in the bucket (issue #461),
//! under the `purges/` sub-prefix, which the walk skips:
//! every replica reads them at boot and on a short refresh,
//! so a purge stays effective on every replica, and across
//! a restart, while its walk runs and for a grace period
//! after.
//!
//! Walks are paced (issue #462): one walk reads
//! [`PURGE_READS_AT_ONCE`] headers at a time, at most
//! [`PURGE_WALKS_AT_ONCE`] walks run in the background at
//! once, the inline parts of all purge requests together
//! keep at most [`PURGE_INLINE_READS_AT_ONCE`] reads in
//! flight, and a purge whose target already has a walk
//! running in this process joins that walk instead of
//! starting another (see [`S3Store::lead_or_join`]). A
//! walk that the store fails part way reports itself
//! incomplete (see [`Swept`]), and its purges are not
//! marked ended.
//!
//! Like the disk store, every operation is best-effort: a
//! network or service error degrades to a miss or a dropped
//! write, never a failed request, because a cache must not
//! take the pipeline down with it. Requests are signed with
//! SigV4 exactly like the source loader's (`sigv4`), with the
//! same credentials (see [`crate::credentials`]): the static
//! `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, plus the
//! optional `AWS_SESSION_TOKEN`, read once from the
//! environment at startup, or, with those unset, the
//! container credentials endpoint's, refreshed before each
//! expiry (issue #709). With no credentials the requests go
//! unsigned, for local S3-compatible development stores.
//!
//! A refusal is not a miss (issue #709): a lookup or an
//! existence check the store answers with an error other than
//! `404`, and a purge read or delete it refuses, count on
//! /metrics by request and in the tier's errors, and are
//! logged at most one line every 10 seconds. A purge walk that
//! meets a refusal has not judged that object, so its purge's
//! tombstone stays in force. A redirect is one of those answers
//! (issue #781): no request follows a `3xx`, so nothing the
//! store is sent goes to the host a redirect names.
//!
//! Walks list the bucket a page at a time (issue #709): one
//! page of at most [`LIST_PAGE_KEYS`] names, judged before the
//! next is listed, so what a walk holds, and what a walk
//! waiting for its turn keeps, is a page and a continuation
//! token, whatever the bucket holds.
//!
//! Every request is bounded (issue #550), by a rule that
//! scales with the bytes it moves instead of one fixed
//! total: see [`transfer_budget`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::cache::{
    ActivePurge, CachedImage, PurgeCutoff, PurgeGuard, PurgeTarget, Tombstone, now_ms,
};
use crate::credentials::AwsCredentials;
use crate::sigv4;
use crate::storage::{
    HeaderProbe, MAX_ENTRY_BYTES, MAX_TOMBSTONE_BYTES, StallLog, key_stem, parse_entry,
    parse_entry_header, parse_tombstone, serialize_entry, serialize_tombstone,
};

/// The sub-prefix purge tombstones live under (issue #461),
/// inside the store prefix. The purge walk never reads it,
/// and entry objects (`<64 hex>.bin`) never fall inside it.
const TOMBSTONE_DIR: &str = "purges/";

/// How many tombstones a refresh keeps, newest first. Purges
/// are operator actions and a tombstone expires a grace
/// period after its walk ends, so the live set stays small;
/// the cap bounds the reads a hostile or neglected prefix
/// can cost a refresh. A tombstone held while a purge's disk
/// files are not deleted (issue #572) does not expire, so
/// enough of them, or a fast run of purges, push older ones
/// past the cap: those are not read and do not apply, and
/// they count as unread (see [`S3Store::tombstones_unread`]).
const MAX_SHARED_TOMBSTONES: usize = 1024;

/// How many tombstones a refresh reads at once, and how many
/// spent ones it deletes at once (issue #549). Each request
/// is one round trip; one after another, a boot over many
/// tombstones took their count times the store's latency,
/// and their count times the 10 s request bound when the
/// store hung. At 16 at a time, the full
/// [`MAX_SHARED_TOMBSTONES`] take 64 round trips, about 3 s
/// at 50 ms each.
const TOMBSTONE_READS_AT_ONCE: usize = 16;

/// The most one tombstone refresh may take, the listing, the
/// reads and the deletes of spent tombstones together (issue
/// #549). Boot serves its first request only after its first
/// refresh, so this bounds how long a slow or hanging store
/// can hold boot. The liveness probe in the Helm chart and
/// the kustomize base restarts a pod that has not answered
/// `/health` by about 45 s (a first probe at 5 s, then one
/// every 20 s, three failures), and the rest of boot must
/// fit in that too. A healthy store finishes far sooner (see
/// [`TOMBSTONE_READS_AT_ONCE`]), and 10 s is
/// [`STORE_TIMEOUT`], so a single hung request fails at
/// about the deadline anyway. What a refresh did not read by
/// then is handled like a failed read, but counted once per
/// refresh (see [`S3Store::refresh_tombstones_within`]).
pub const TOMBSTONE_REFRESH_DEADLINE: Duration = Duration::from_secs(10);

/// How many entry headers one purge walk reads at once
/// (issue #462). Each read is a ranged GET, one round trip
/// to the store, so a walk over N objects takes about
/// N x RTT / 16 instead of N x RTT, and one walk still
/// keeps no more than this many requests open.
pub const PURGE_READS_AT_ONCE: usize = 16;

/// How many purge walks run in the background at once
/// (issue #462); the others wait their turn, first come
/// first served. A walk lists the whole prefix, a page at a
/// time (issue #709), and reads every entry header, so walks
/// side by side only split the store's request rate between
/// them, while each one keeps its purge running (its
/// tombstone open) until it ends. A walk waiting for its
/// turn holds a continuation token, not the listing. A
/// purge's inline part (issue #368) is bounded on its own and
/// never waits for a turn (see
/// [`PURGE_INLINE_READS_AT_ONCE`]).
pub const PURGE_WALKS_AT_ONCE: usize = 2;

/// How many entry header reads the inline parts of all of
/// this process's purge requests keep in flight together
/// (issue #462), first come first served. Walk turns bound
/// the background walks only, so without this N concurrent
/// purges of different targets would put N times
/// [`PURGE_READS_AT_ONCE`] reads on the store at once. The
/// background walks do not draw on it, so a request's scan
/// never waits behind them: a process keeps at most this
/// many purge reads in flight plus [`PURGE_READS_AT_ONCE`]
/// for each of the [`PURGE_WALKS_AT_ONCE`] walks.
pub const PURGE_INLINE_READS_AT_ONCE: usize = 2 * PURGE_READS_AT_ONCE;

/// The short bound on a store exchange (issue #550). A
/// request with no body must have its answer begin within
/// it, and no response body may go this long without a
/// byte. A request that moves no body (a HEAD, a DELETE, a
/// listing page) is done within it, as every request was
/// before, and so, give or take its size, is a small object
/// read. It is also the base of every [`transfer_budget`].
const STORE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long connecting to the store may take, inside the
/// request's own bound.
const STORE_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The slowest transfer a store body is allowed (issue
/// #550), in bytes a second: one MiB, so a body of N MiB
/// gets N seconds on top of [`STORE_TIMEOUT`].
const STORE_MIN_RATE: u64 = 1024 * 1024;

/// How long moving `bytes` to or from the store may take
/// (issue #550): [`STORE_TIMEOUT`], plus the bytes at
/// [`STORE_MIN_RATE`]. A write gets it for its body from
/// the moment it starts, answer included; an object read
/// gets it for the length the store declares, counted from
/// the request, or, when none is declared, for what has
/// arrived so far. Bodies are capped (an entry at
/// [`MAX_ENTRY_BYTES`], 512 MiB; an upload at the source
/// fetch cap, at most 1 GiB), so the budget is too: 8 min
/// 42 s for the largest entry, 17 min 14 s for the largest
/// upload. A store that stops reading or sending fails the
/// request at that budget; one that answers nothing to a
/// request with no body fails it at [`STORE_TIMEOUT`].
///
/// Before, every request had one 10 s total, so any body
/// that took longer to move failed however steadily it
/// went, although the caps allow up to 1 GiB.
fn transfer_budget(bytes: u64) -> Duration {
    STORE_TIMEOUT + Duration::from_millis(bytes.saturating_mul(1000) / STORE_MIN_RATE)
}

/// A request body and the lower-case hex SHA-256 its
/// signature names (`x-amz-content-sha256`).
struct Payload {
    bytes: Vec<u8>,
    sha256: String,
}

impl Payload {
    /// No body: the digest of nothing is a constant (issue
    /// #644).
    fn empty() -> Self {
        Payload {
            bytes: Vec::new(),
            sha256: sigv4::EMPTY_PAYLOAD_SHA256.to_string(),
        }
    }

    /// `bytes` with their digest, taken here.
    fn hashed(bytes: Vec<u8>) -> Self {
        let sha256 = hex::encode(Sha256::digest(&bytes));
        Payload { bytes, sha256 }
    }

    /// `bytes` with their digest, taken on the blocking pool
    /// when they pass [`INLINE_HASH_BYTES`] (issue #709): a
    /// body can be a 512 MiB entry or a 1 GiB upload. The
    /// digest is the same either way. `None` when that task
    /// did not report (the runtime is shutting down).
    async fn of(bytes: Vec<u8>) -> Option<Self> {
        if bytes.is_empty() {
            return Some(Self::empty());
        }
        if bytes.len() <= INLINE_HASH_BYTES {
            return Some(Self::hashed(bytes));
        }
        tokio::task::spawn_blocking(move || Self::hashed(bytes))
            .await
            .ok()
    }
}

/// How [`S3Store::send`] bounds one request (issue #550).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    /// The whole exchange, response body included, within
    /// the [`transfer_budget`] of the request body: a write
    /// scales with what it sends, and a request with no
    /// body keeps [`STORE_TIMEOUT`].
    Exchange,
    /// An object read, whose body may be anything up to the
    /// entry cap: the answer must begin within
    /// [`STORE_TIMEOUT`] (the client's read timeout), and
    /// the caller reads the body under the budget of its
    /// declared length ([`read_capped_at`]).
    ObjectRead,
}

/// The most names one listing page asks for: S3's own cap,
/// and its default (issue #709). A purge walk judges a page
/// before it lists the next.
pub const LIST_PAGE_KEYS: usize = 1000;

/// The most of a listing page's body a walk reads: a page of
/// [`LIST_PAGE_KEYS`] names of S3's longest, 1024 bytes, each
/// of which the XML may spell in as many as six (`&quot;`),
/// is about 6.6 MB with the rest of each entry's fields.
const LIST_PAGE_MAX_BYTES: usize = 8 * 1024 * 1024;

/// How much of an error answer's body a lookup reads for its
/// error code (issue #709): S3 puts `<Code>` first, and the
/// rest of a document can run past this, as a
/// `SignatureDoesNotMatch` that spells out a canonical request
/// with a session token does.
const REFUSAL_BODY_BYTES: usize = 4 * 1024;

/// A body past this many bytes is hashed on the blocking
/// pool, not on a runtime worker (issue #709): SHA-256 runs at
/// a few hundred MB a second, so a large entry or upload held
/// a worker for as long as a second, and every task queued on
/// it waited. Below it, the hash takes tens of microseconds,
/// less than the hand-off.
const INLINE_HASH_BYTES: usize = 16 * 1024;

/// What the requests a rate-limited line leaves out are, after
/// their count (issue #709).
const REFUSAL_LOG_REST: &str = "refused or failed requests";

/// The kinds of shared-tier request a refusal counts under
/// (issue #709), as `crabcrop_cache_shared_refused_total{request}`.
/// A refusal is an error answer that is not a store failure
/// (see [`store_failed`]) and not one that says there is
/// nothing to act on, such as a `404`: the `403` of
/// credentials the bucket does not honor, say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// A lookup's GET.
    Lookup,
    /// A purge walk's read of an entry header, or its delete
    /// of a covered copy.
    Purge,
}

impl Refused {
    pub const ALL: [Refused; 2] = [Refused::Lookup, Refused::Purge];

    /// The `request` label value.
    pub fn label(self) -> &'static str {
        match self {
            Refused::Lookup => "lookup",
            Refused::Purge => "purge",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The outcome of a conditional raw write
/// ([`S3Store::put_raw_if_absent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawPut {
    /// The object was stored.
    Stored,
    /// An object already existed under the name; nothing
    /// was written.
    AlreadyExists,
    /// The store refused or could not be reached.
    Failed,
}

pub struct S3Store {
    /// The client for requests with no body: a read timeout
    /// of [`STORE_TIMEOUT`] bounds the wait for an answer
    /// and every stall in a response body (issue #550).
    client: reqwest::Client,
    /// The client for writes, with no read timeout: reqwest
    /// starts that timeout with the request and does not
    /// reset it while the body goes out, so it would fail a
    /// write that takes longer however steadily it moved.
    /// Each write runs under its [`transfer_budget`].
    put_client: reqwest::Client,
    bucket: String,
    /// Normalized object prefix: empty, or ending in `/`.
    prefix: String,
    endpoint: Option<String>,
    region: String,
    creds: Option<AwsCredentials>,
    /// The cache entry MAC key, derived from the server
    /// secret (issue #190): entries are written and
    /// verified under it, so an object planted in the
    /// bucket by anyone without the secret is a miss.
    /// Inert for the raw-object path (`put_raw`/`get_raw`
    /// store uploads verbatim, with no entry framing).
    /// Zeroed when the store is dropped.
    entry_key: zeroize::Zeroizing<Vec<u8>>,
    /// Largest object an entry read will buffer. Objects in
    /// the shared bucket are operator-adjacent data, not
    /// fully trusted input: a planted oversized object must
    /// not turn a lookup or a purge walk into an unbounded
    /// allocation.
    max_entry_bytes: usize,
    /// The tombstones the last refresh read (issue #461),
    /// by object name, with the ETag the listing gave each:
    /// a refresh fetches only the objects that are new or
    /// changed since.
    tombstones: std::sync::Mutex<HashMap<String, (String, Tombstone)>>,
    /// How many purge tombstones the last complete listing
    /// named that this replica has never read (issue #577):
    /// their reads failed or did not finish, and no earlier
    /// version is in force, or they were listed past the newest
    /// [`MAX_SHARED_TOMBSTONES`] (issue #572). Their purges do
    /// not apply here until a refresh reads them. Surfaced on
    /// /metrics as `crabcrop_cache_shared_tombstones_unread`.
    tombstones_unread: std::sync::atomic::AtomicU64,
    /// When this replica last listed the tombstones
    /// completely, in Unix ms; 0 before its first complete
    /// listing (issue #577). A purge another replica began
    /// after it has not reached this replica's guard.
    /// Surfaced on /metrics as
    /// `crabcrop_cache_shared_tombstones_listed_timestamp_seconds`.
    tombstones_listed_ms: std::sync::atomic::AtomicU64,
    /// The purge walks running in this process, by target,
    /// each with the identical purges that joined it and
    /// wait for its re-list (issue #462).
    walks: std::sync::Mutex<HashMap<PurgeTarget, Vec<SharedPurge>>>,
    /// The purges in this process whose disk files are not all
    /// deleted yet, by target, with the tombstones of that
    /// target waiting for them before they are marked ended
    /// (issue #572).
    held_ends: std::sync::Mutex<HashMap<PurgeTarget, HeldEnds>>,
    /// Background walk turns (issue #462):
    /// [`PURGE_WALKS_AT_ONCE`] permits, granted in arrival
    /// order.
    walk_turns: tokio::sync::Semaphore,
    /// Inline purge header reads in flight (issue #462):
    /// [`PURGE_INLINE_READS_AT_ONCE`] permits, one per read.
    inline_reads: tokio::sync::Semaphore,
    /// Tier failures, counted and logged as they happen
    /// (issue #367): the shared tier stays best-effort,
    /// but a store that rejects every write must be
    /// visible in the logs and on /metrics instead of
    /// failing silently forever.
    errors: std::sync::atomic::AtomicU64,
    /// Requests the store refused, by [`Refused`] kind (issue
    /// #709): a lookup's also counts in `errors`, and a purge
    /// walk's in the one error its pass counts.
    refused: [std::sync::atomic::AtomicU64; 2],
    /// Existence checks that failed (issue #709): refused, a
    /// store failure, or no answer. Each also counts in
    /// `errors`. Only the upload bucket's store makes them.
    exists_failed: std::sync::atomic::AtomicU64,
    /// What the store's log lines start with: `shared storage`
    /// for the result tier, `upload storage` for the upload
    /// bucket (see [`S3Store::logging_as`]).
    tier: &'static str,
    /// The rate limit on the lines refused, failed and
    /// unanswered lookups and existence checks log (issue
    /// #709): a store that refuses every lookup refuses one
    /// per request.
    refusal_log: StallLog,
}

impl std::fmt::Debug for S3Store {
    /// The entry MAC key prints as `[redacted]`, and the
    /// credentials redact their own secret (issue #455).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Store")
            .field("client", &self.client)
            .field("put_client", &self.put_client)
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("endpoint", &self.endpoint)
            .field("region", &self.region)
            .field("creds", &self.creds)
            .field("entry_key", &crate::redact::Redacted)
            .field("max_entry_bytes", &self.max_entry_bytes)
            .field("errors", &self.errors)
            .field("refused", &self.refused)
            .field("exists_failed", &self.exists_failed)
            .field("tier", &self.tier)
            .finish()
    }
}

impl S3Store {
    /// Fails when the HTTP client cannot be built (no CA
    /// roots on the host, say). Boot reports the error
    /// instead of falling back to a default client, which
    /// would panic on the same failure or else run with no
    /// request timeout (issue #455).
    pub fn new(
        bucket: String,
        prefix: String,
        endpoint: Option<String>,
        region: String,
        creds: Option<AwsCredentials>,
        entry_key: Vec<u8>,
    ) -> Result<Self, reqwest::Error> {
        let trimmed = prefix.trim_matches('/');
        let prefix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("{trimmed}/")
        };
        Ok(Self {
            // No client-wide total (issue #550): every request
            // gets its own bound, sized to what it moves (see
            // `send`). Neither client follows a redirect (issue
            // #781), as the loader's store client and the
            // container credentials client do not: a followed
            // `3xx` carried the session token, the date and a
            // write's body to whatever host it named. A `3xx`
            // is the store's answer, refused like a `403`.
            client: reqwest::Client::builder()
                .connect_timeout(STORE_CONNECT_TIMEOUT)
                .read_timeout(STORE_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            put_client: reqwest::Client::builder()
                .connect_timeout(STORE_CONNECT_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            bucket,
            prefix,
            endpoint,
            region,
            creds,
            entry_key: zeroize::Zeroizing::new(entry_key),
            tombstones: std::sync::Mutex::new(HashMap::new()),
            tombstones_unread: std::sync::atomic::AtomicU64::new(0),
            tombstones_listed_ms: std::sync::atomic::AtomicU64::new(0),
            walks: std::sync::Mutex::new(HashMap::new()),
            held_ends: std::sync::Mutex::new(HashMap::new()),
            walk_turns: tokio::sync::Semaphore::new(PURGE_WALKS_AT_ONCE),
            inline_reads: tokio::sync::Semaphore::new(PURGE_INLINE_READS_AT_ONCE),
            max_entry_bytes: MAX_ENTRY_BYTES,
            errors: std::sync::atomic::AtomicU64::new(0),
            refused: Default::default(),
            exists_failed: std::sync::atomic::AtomicU64::new(0),
            tier: "shared storage",
            refusal_log: StallLog::for_tier("shared storage", REFUSAL_LOG_REST),
        })
    }

    /// The store with its log lines starting `tier` instead of
    /// `shared storage`: the upload bucket's store is not the
    /// shared result tier, and its lines must not read as if it
    /// were (issue #709).
    pub fn logging_as(mut self, tier: &'static str) -> Self {
        self.tier = tier;
        self.refusal_log = StallLog::for_tier(tier, REFUSAL_LOG_REST);
        self
    }

    /// Tier failures so far (issue #367), surfaced on
    /// /metrics as `crabcrop_cache_tier_errors_total`.
    pub fn errors(&self) -> u64 {
        self.errors.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The listed tombstones this replica has never read, as
    /// of its last complete listing (issue #577; see the
    /// field). Relaxed: a scrape reads a recent value.
    pub fn tombstones_unread(&self) -> u64 {
        self.tombstones_unread
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// When this replica last listed the tombstones
    /// completely, in Unix ms, or 0 (issue #577; see the
    /// field).
    pub fn tombstones_listed_ms(&self) -> u64 {
        self.tombstones_listed_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Count and log one tier failure. The operation
    /// keeps its best-effort result (a dropped write, a
    /// miss), but never silently (issue #367).
    fn note_error(&self, what: &str, detail: &str) {
        self.note_errors(1, what, detail);
    }

    /// Count `count` tier failures that one line reports
    /// together, and log that line.
    fn note_errors(&self, count: u64, what: &str, detail: &str) {
        self.errors
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        stderr_line!("{}: {what}: {detail}", self.tier);
    }

    /// Requests of `kind` the store has refused so far (issue
    /// #709), surfaced on /metrics as
    /// `crabcrop_cache_shared_refused_total`.
    pub fn refused(&self, kind: Refused) -> u64 {
        self.refused[kind.index()].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Existence checks that failed so far (issue #709): the
    /// upload bucket's store surfaces them on /metrics as
    /// `crabcrop_upload_exists_checks_failed_total`.
    pub fn exists_checks_failed(&self) -> u64 {
        self.exists_failed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Count and log one refusal of a request of `kind` (issue
    /// #709): `status`, and the error code the store gave, if
    /// it gave one. A refused lookup counts in the tier's
    /// errors too, as one that got no answer does; a purge
    /// walk's refusals count there once per pass, with its
    /// other failures (see [`S3Store::note_pass`]).
    fn note_refused(
        &self,
        kind: Refused,
        what: &str,
        status: reqwest::StatusCode,
        code: Option<&str>,
    ) {
        self.refused[kind.index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if kind != Refused::Purge {
            self.errors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.log_answer("refused", what, status, code);
    }

    /// Count and log one request the store answered with a
    /// store failure (issue #709): a server error, throttling
    /// or a timeout (see [`store_failed`]). It counts in the
    /// tier's errors, not among the refusals.
    fn note_failed(&self, what: &str, status: reqwest::StatusCode, code: Option<&str>) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.log_answer("failed", what, status, code);
    }

    /// Log `the store <verb> <what>: <status> (<code>)` like the
    /// disk tier's stalls: the first at once, then at most one
    /// line every 10 seconds with a count of the rest.
    fn log_answer(&self, verb: &str, what: &str, status: reqwest::StatusCode, code: Option<&str>) {
        let line = match code {
            Some(code) => format!("the store {verb} {what}: {status} ({code})"),
            None => format!("the store {verb} {what}: {status}"),
        };
        if let Some(line) = self.refusal_log.note_now(&line) {
            stderr_line!("{line}");
        }
    }

    /// Count and log, at the same rate, one request the store
    /// did not answer (issue #709).
    fn note_unanswered(&self, what: &str) {
        self.errors
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(line) = self.refusal_log.note_now(&format!(
            "cannot reach the store for {what}: transport error"
        )) {
            stderr_line!("{line}");
        }
    }

    /// Read a response body under the entry size ceiling,
    /// streaming so an oversized object is refused as soon
    /// as it passes the cap instead of after it has been
    /// buffered whole, and within the transfer budget from
    /// `started`, when its request began (issue #550). An
    /// oversized body, one that breaks off, and one past its
    /// budget all read as a cache miss upstream.
    async fn read_capped(
        &self,
        resp: reqwest::Response,
        started: Instant,
    ) -> Result<Vec<u8>, BodyFailure> {
        read_capped_at(resp, self.max_entry_bytes, started).await
    }

    fn object_key(&self, stem: &str) -> String {
        format!("{}{stem}.bin", self.prefix)
    }

    /// One signed (or unsigned) request against the bucket.
    /// `object_key` empty addresses the bucket itself (LIST).
    /// Returns the response, or None on any transport error:
    /// callers read every failure as a cache miss.
    async fn request(
        &self,
        method: &str,
        object_key: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Option<reqwest::Response> {
        self.request_h(method, object_key, query, body, &[]).await
    }

    /// [`S3Store::request`] with extra request headers (a
    /// write's precondition and content type, a delete's
    /// `If-Match`).
    async fn request_h(
        &self,
        method: &str,
        object_key: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
        extra_headers: &[(&str, &str)],
    ) -> Option<reqwest::Response> {
        let payload = Payload::of(body).await?;
        self.send(
            method,
            object_key,
            query,
            payload,
            extra_headers,
            Bound::Exchange,
        )
        .await
    }

    /// A GET of one object whose body the caller reads with
    /// [`read_capped_at`], under the budget of the length
    /// the store declares (issue #550), from the `Instant`
    /// it took before calling: an entry, a header probe
    /// (with its `Range`), a tombstone, a raw object.
    async fn request_object(
        &self,
        object_key: &str,
        extra_headers: &[(&str, &str)],
    ) -> Option<reqwest::Response> {
        self.send(
            "GET",
            object_key,
            &[],
            Payload::empty(),
            extra_headers,
            Bound::ObjectRead,
        )
        .await
    }

    /// Sign (when there are credentials) and send one
    /// request, bounded as `bound` says (issue #550).
    async fn send(
        &self,
        method: &str,
        object_key: &str,
        query: &[(&str, &str)],
        payload: Payload,
        extra_headers: &[(&str, &str)],
        bound: Bound,
    ) -> Option<reqwest::Response> {
        let path_part = if object_key.is_empty() {
            String::new()
        } else {
            format!("/{}", sigv4::encode_path(object_key))
        };
        let mut url_string = match &self.endpoint {
            Some(ep) => format!("{ep}/{}{path_part}", self.bucket),
            None => format!(
                "https://{}.s3.{}.amazonaws.com{path_part}",
                self.bucket, self.region
            ),
        };
        if !query.is_empty() {
            let qs: Vec<String> = query
                .iter()
                .map(|(k, v)| format!("{}={}", sigv4::uri_encode(k), sigv4::uri_encode(v)))
                .collect();
            url_string = format!("{url_string}?{}", qs.join("&"));
        }
        let url = reqwest::Url::parse(&url_string).ok()?;
        // The caller hashed the body (see `Payload`): a read's
        // digest is the constant of nothing (issue #644), and a
        // large write's was taken off the runtime (issue #709).
        let Payload {
            bytes: body,
            sha256: payload_hash,
        } = payload;
        let verb = reqwest::Method::from_bytes(method.as_bytes()).ok()?;
        // A body goes out on the client without a read
        // timeout, under its own budget (issue #550).
        let (client, budget) = if body.is_empty() {
            (&self.client, transfer_budget(0))
        } else {
            (&self.put_client, transfer_budget(body.len() as u64))
        };
        let mut request = client
            .request(verb, url.clone())
            .header("x-amz-content-sha256", payload_hash.clone())
            .body(body);
        if bound == Bound::Exchange {
            request = request.timeout(budget);
        }
        for (name, value) in extra_headers {
            request = request.header(*name, *value);
        }
        // The credentials in force now: a container source's
        // refresh replaces them before they expire (issue #709).
        if let Some(creds) = self.creds.as_ref().and_then(AwsCredentials::current) {
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
                ("x-amz-content-sha256", payload_hash.as_str()),
                ("x-amz-date", amz_date.as_str()),
            ];
            if let Some(token) = &creds.session_token {
                sign_headers.push(("x-amz-security-token", token.as_str()));
            }
            let authorization = sigv4::authorization(
                &creds,
                &sigv4::SignRequest {
                    method,
                    path: url.path(),
                    query,
                    headers: &sign_headers,
                    region: &self.region,
                    service: "s3",
                    payload_sha256: &payload_hash,
                    amz_date: &amz_date,
                },
            );
            request = request
                .header("Authorization", authorization)
                .header("x-amz-date", amz_date);
            if let Some(token) = &creds.session_token {
                request = request.header("x-amz-security-token", token.as_str());
            }
        }
        request.send().await.ok()
    }

    /// Look up `key`. Any failure (missing object, unreadable
    /// entry, transport error) is a miss. A transport failure
    /// is counted and logged (issue #367): a missing object
    /// is an ordinary miss, but a store that cannot be
    /// reached at all must not look like one. Neither must
    /// one that refuses (issue #709): any answer but a success
    /// or a `404`, such as the `403` of expired or missing
    /// credentials, is counted as a refused lookup and logged
    /// with the store's error code, at a limited rate.
    pub async fn get(&self, key: &str) -> Option<CachedImage> {
        self.get_tagged(key).await.map(|(image, _)| image)
    }

    /// [`S3Store::get`], plus the object's ETag as the store
    /// reported it, so a caller that decides to delete the
    /// copy it read can delete exactly that copy (issue
    /// #446) with [`S3Store::remove_if_match`].
    pub async fn get_tagged(&self, key: &str) -> Option<(CachedImage, Option<String>)> {
        let object = self.object_key(&key_stem(key));
        let started = Instant::now();
        let resp = match self.request_object(&object, &[]).await {
            Some(resp) => resp,
            None => {
                self.note_error("cannot reach the store for a lookup", "transport error");
                return None;
            }
        };
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return None;
        }
        if !status.is_success() {
            let code = refusal_code(resp, started).await;
            if store_failed(status) {
                self.note_failed("a lookup", status, code.as_deref());
            } else {
                self.note_refused(Refused::Lookup, "a lookup", status, code.as_deref());
            }
            return None;
        }
        let etag = response_etag(&resp);
        let data = match self.read_capped(resp, started).await {
            Ok(data) => data,
            Err(BodyFailure::TooLarge) => return None,
            // A body that broke off or ran past its budget is
            // a store failure, counted like one that never
            // answered (issues #367, #550), not a plain miss.
            Err(BodyFailure::Failed) => {
                self.note_error(
                    "cannot read an entry",
                    "transport error or past its time bound",
                );
                return None;
            }
        };
        let (stored_key, image) = parse_entry(&data, &self.entry_key)?;
        if stored_key != key {
            return None;
        }
        Some((image, etag))
    }

    /// Store `image` under `key`. The write stays
    /// best-effort (the local tiers already hold the
    /// entry), but a failure, transport or status, is
    /// counted and logged (issue #367).
    pub async fn put(&self, key: &str, image: &CachedImage) {
        let object = self.object_key(&key_stem(key));
        let Some(payload) = self.entry_payload(key, image).await else {
            self.note_error("cannot store an entry", "its serialization task failed");
            return;
        };
        let sent = self
            .send("PUT", &object, &[], payload, &[], Bound::Exchange)
            .await;
        match sent {
            Some(resp) if resp.status().is_success() => {}
            Some(resp) => self.note_error(
                "cannot store an entry",
                &format!("status {}", resp.status()),
            ),
            None => self.note_error("cannot store an entry", "transport error"),
        }
    }

    /// `image` serialized as the entry for `key`, with the
    /// hash its write signs (issue #709). Serializing copies
    /// and MACs every byte, and hashing reads them all again,
    /// so a large image does both on the blocking pool, off
    /// the runtime worker. `None` when that task did not report
    /// (the runtime is shutting down).
    async fn entry_payload(&self, key: &str, image: &CachedImage) -> Option<Payload> {
        if image.bytes.len() <= INLINE_HASH_BYTES {
            return Some(Payload::hashed(serialize_entry(
                key,
                image,
                &self.entry_key,
            )));
        }
        let (key, image, entry_key) = (key.to_string(), image.clone(), self.entry_key.clone());
        tokio::task::spawn_blocking(move || {
            Payload::hashed(serialize_entry(&key, &image, &entry_key))
        })
        .await
        .ok()
    }

    /// Delete the single entry stored under `key`, if any:
    /// the one-object counterpart of the purge walk, for a
    /// caller that knows the exact key it wrote.
    pub async fn remove(&self, key: &str) {
        let object = self.object_key(&key_stem(key));
        self.delete_object(&object).await;
    }

    /// Delete the entry stored under `key` only while the
    /// object is still the one whose ETag is `etag` (issue
    /// #446): a copy a lookup found purged is removed, but a
    /// fresh render that replaced it since is left alone.
    /// Without an ETag the delete is unconditional. A store
    /// that ignores the precondition, or refuses it as not
    /// implemented, deletes unconditionally, the behavior
    /// before it.
    pub async fn remove_if_match(&self, key: &str, etag: Option<&str>) {
        let object = self.object_key(&key_stem(key));
        self.delete_object_if_match(&object, etag).await;
    }

    /// Store raw bytes under `name`, verbatim (issue #168):
    /// the object key is the store prefix plus `name`, with
    /// no cache-entry serialization and no `.bin` suffix.
    /// This is the upload destination's write path, so
    /// unlike the cache `put` it reports failure: an upload
    /// that did not land must not answer as if it did.
    /// Returns whether the store accepted the object.
    pub async fn put_raw(&self, name: &str, bytes: Vec<u8>) -> bool {
        let object = format!("{}{name}", self.prefix);
        // The object carries the content type the bytes
        // sniff as (issue #366): the loader admits store
        // responses by content type, so an object stored
        // without one is only as readable as the store's
        // default for headerless objects.
        let headers: Vec<(&str, &str)> = match crate::engine::sniff_content_type(&bytes) {
            Some(content_type) => vec![("content-type", content_type)],
            None => Vec::new(),
        };
        match self.request_h("PUT", &object, &[], bytes, &headers).await {
            Some(resp) => resp.status().is_success(),
            None => false,
        }
    }

    /// Read the raw bytes stored under `name` (see
    /// [`S3Store::put_raw`]). Any failure (missing object,
    /// transport error) is `None`. The read half of the
    /// raw pair, which no request path calls: the upload
    /// existence check goes through [`S3Store::exists_raw`],
    /// a HEAD (issue #192), and serving reads of uploaded
    /// objects go through the loader.
    pub async fn get_raw(&self, name: &str) -> Option<Vec<u8>> {
        let object = format!("{}{name}", self.prefix);
        let started = Instant::now();
        let resp = self.request_object(&object, &[]).await?;
        if !resp.status().is_success() {
            return None;
        }
        // Uncapped: an upload's own cap bounded what was
        // stored. The budget still applies (issue #550).
        read_capped_at(resp, usize::MAX, started).await.ok()
    }

    /// Whether a raw object exists under `name`, answered
    /// with a HEAD: no body is read (issue #192, audit
    /// R2-23). The upload endpoint's existence check runs
    /// on every signed PUT; the object it asks about can
    /// be arbitrarily large, and downloading it whole to
    /// answer a boolean made the check a bandwidth and
    /// memory sink.
    pub async fn exists_raw(&self, name: &str) -> bool {
        let object = format!("{}{name}", self.prefix);
        self.object_exists(&object).await
    }

    /// Store raw bytes under `name` only if no object is
    /// there yet (issue #192, audit R2-21): a conditional
    /// PUT (`If-None-Match: *`), so the existence decision
    /// and the write are one atomic step instead of a
    /// check-then-write pair another upload can slip
    /// between. Stores without conditional-write support
    /// degrade to an unconditional write; Amazon S3 and
    /// current S3-compatible stores honor the precondition
    /// and answer `412` for a name already taken.
    pub async fn put_raw_if_absent(&self, name: &str, bytes: Vec<u8>) -> RawPut {
        let object = format!("{}{name}", self.prefix);
        // The content type rides along with the conditional
        // write, as in `put_raw` (issue #366).
        let mut headers: Vec<(&str, &str)> = vec![("if-none-match", "*")];
        if let Some(content_type) = crate::engine::sniff_content_type(&bytes) {
            headers.push(("content-type", content_type));
        }
        match self.request_h("PUT", &object, &[], bytes, &headers).await {
            Some(resp) if resp.status().is_success() => RawPut::Stored,
            Some(resp) if resp.status() == reqwest::StatusCode::PRECONDITION_FAILED => {
                RawPut::AlreadyExists
            }
            _ => RawPut::Failed,
        }
    }

    /// One page of the entry objects under the store prefix,
    /// from `token` (`None`: the first page), of at most
    /// `max_keys` names when given and the store's page
    /// otherwise: the purge walk's input, a page at a time
    /// (issue #709). `None` when the store did not list it.
    ///
    /// A page lists every object under the prefix, but only
    /// names of the entry form are kept (see
    /// [`S3Store::is_entry_object`]). Tombstones (issue
    /// #461), raw uploads, and the objects of a deployment
    /// whose prefix sits inside this one are never read
    /// (issue #608). The names are filtered here, not with a
    /// `delimiter` in the request, so this works on every
    /// S3-compatible store.
    async fn list_entry_page(
        &self,
        token: Option<&str>,
        max_keys: Option<usize>,
    ) -> Option<EntryPage> {
        let page = self.list_page(&self.prefix, token, max_keys).await?;
        let listed = page.entries.len();
        let objects = page
            .entries
            .into_iter()
            .map(|(key, _)| key)
            .filter(|key| self.is_entry_object(key))
            .collect();
        Some(EntryPage {
            objects,
            listed,
            next: page.next,
        })
    }

    /// Whether `object` has the form of an entry object this
    /// store writes (issue #608): the store prefix, then a key
    /// stem (64 lowercase hex digits), then `.bin`. An object
    /// under a nested prefix has a `/` after the store prefix,
    /// so it fails the form.
    fn is_entry_object(&self, object: &str) -> bool {
        object
            .strip_prefix(&self.prefix)
            .and_then(|name| name.strip_suffix(".bin"))
            .is_some_and(|stem| {
                stem.len() == 64 && stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            })
    }

    /// Every (object key, ETag) under `prefix`, following
    /// continuation tokens, and whether the listing ran to
    /// its end. An error stops the listing where it is. The
    /// tombstone refresh's listing: a few objects, read whole.
    async fn list_under(&self, prefix: &str) -> (Vec<(String, String)>, bool) {
        let mut listed = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let Some(page) = self.list_page(prefix, token.as_deref(), None).await else {
                return (listed, false);
            };
            listed.extend(page.entries);
            match page.next {
                None => return (listed, true),
                next => token = next,
            }
        }
    }

    /// One ListObjectsV2 page of `prefix`: from `token` (`None`:
    /// the first page), of at most `max_keys` names when given.
    /// `None` when the store did not answer it, answered with
    /// an error, or answered a truncated page with no token to
    /// go on from, or with the token it was asked with, which
    /// would list the same page forever.
    async fn list_page(
        &self,
        prefix: &str,
        token: Option<&str>,
        max_keys: Option<usize>,
    ) -> Option<ListPage> {
        let max_keys = max_keys.map(|n| n.clamp(1, LIST_PAGE_KEYS).to_string());
        let mut query: Vec<(&str, &str)> = vec![("list-type", "2")];
        if !prefix.is_empty() {
            query.push(("prefix", prefix));
        }
        if let Some(t) = token {
            query.push(("continuation-token", t));
        }
        if let Some(n) = &max_keys {
            query.push(("max-keys", n));
        }
        let started = Instant::now();
        let resp = self.request("GET", "", &query, Vec::new()).await?;
        if !resp.status().is_success() {
            return None;
        }
        let body = read_capped_at(resp, LIST_PAGE_MAX_BYTES, started)
            .await
            .ok()?;
        let text = String::from_utf8(body).ok()?;
        let entries = text
            .split("<Contents>")
            .skip(1)
            .filter_map(|block| {
                let key = xml_field(block, "Key")?;
                Some((key, xml_field(block, "ETag").unwrap_or_default()))
            })
            .collect();
        if !text.contains("<IsTruncated>true</IsTruncated>") {
            return Some(ListPage {
                entries,
                next: None,
            });
        }
        let next = xml_field(&text, "NextContinuationToken").filter(|t| !t.is_empty())?;
        if token == Some(next.as_str()) {
            return None;
        }
        Some(ListPage {
            entries,
            next: Some(next),
        })
    }

    /// How much of an object the purge walk reads to learn
    /// its stored key and groups: the header (field lines,
    /// the key, the groups block) sits at the front of every
    /// entry, and real keys and group lists are far smaller
    /// than this. A header that outgrows the probe falls
    /// back to one full read of that object.
    const HEADER_PROBE_BYTES: usize = 16 * 1024;

    /// Read one object's stored key and groups from its
    /// header alone, via a ranged GET (audit R2-24: so a
    /// purge does not download every entry in the tier,
    /// image bytes included). The header is a match hint, not a
    /// serving read: it is parsed without MAC verification,
    /// because verification needs the whole object. A
    /// forged header can at worst steer a purge, and only to
    /// the object stored under its own key's name (issue
    /// #608); entries are only ever SERVED through `get`,
    /// which verifies.
    /// Objects past the entry size ceiling are refused
    /// outright, exactly as `get` refuses them.
    ///
    /// A read the store did not answer is told apart from an
    /// object with nothing to purge in it (issue #462): the
    /// walk that meets one has not judged that object.
    async fn read_entry_header(&self, object: &str) -> HeaderRead {
        let range = format!("bytes=0-{}", Self::HEADER_PROBE_BYTES - 1);
        let started = Instant::now();
        let Some(resp) = self
            .request_object(object, &[("range", range.as_str())])
            .await
        else {
            return HeaderRead::Failed;
        };
        if !resp.status().is_success() {
            return HeaderRead::of_error(resp, started).await;
        }
        // A ranged GET reports the whole object's ETag.
        let etag = response_etag(&resp);
        let partial = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        // The object's real length: from Content-Range on a
        // partial response, else the response's own length.
        let Some(total) = resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit('/').next())
            .and_then(|v| v.parse::<u64>().ok())
            .or_else(|| resp.content_length())
        else {
            return HeaderRead::Nothing;
        };
        if total > self.max_entry_bytes as u64 {
            return HeaderRead::Nothing;
        }
        // The length was checked above, so a body that does
        // not arrive whole is a transport failure. A store
        // that ignores the range sends the whole object, and
        // its budget follows that length (issue #550).
        let Ok(data) = self.read_capped(resp, started).await else {
            return HeaderRead::Failed;
        };
        let whole = |data: &[u8], etag: Option<String>| match parse_entry(data, &self.entry_key) {
            Some((key, image)) => HeaderRead::Entry(ProbedEntry {
                header: HeaderProbe {
                    key,
                    groups: image.groups,
                    written: image.written,
                },
                etag,
            }),
            None => HeaderRead::Nothing,
        };
        if !partial || data.len() as u64 >= total {
            // The whole object arrived (the store ignored
            // the range, or the entry fits in the probe).
            return whole(&data, etag);
        }
        match parse_entry_header(&data, total as usize) {
            Some(header) => HeaderRead::Entry(ProbedEntry { header, etag }),
            None => {
                // The header outgrew the probe: one full
                // read of this object settles it.
                let started = Instant::now();
                let Some(resp) = self.request_object(object, &[]).await else {
                    return HeaderRead::Failed;
                };
                if !resp.status().is_success() {
                    return HeaderRead::of_error(resp, started).await;
                }
                let etag = response_etag(&resp);
                match self.read_capped(resp, started).await {
                    Ok(data) => whole(&data, etag),
                    Err(_) => HeaderRead::Failed,
                }
            }
        }
    }

    /// Whether an object exists (a HEAD, no body read). A
    /// HEAD that fails reads as no object, as before, but is
    /// counted and logged at a limited rate (issue #709):
    /// refused, failed or unanswered, it counts among the
    /// failed existence checks and in the store's errors.
    async fn object_exists(&self, object: &str) -> bool {
        let answer = self.request("HEAD", object, &[], Vec::new()).await;
        if let Some(resp) = &answer {
            if resp.status().is_success() {
                return true;
            }
            if resp.status() == reqwest::StatusCode::NOT_FOUND {
                return false;
            }
        }
        self.exists_failed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match answer {
            Some(resp) if store_failed(resp.status()) => {
                self.note_failed("an existence check", resp.status(), None);
            }
            Some(resp) => {
                self.errors
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.log_answer("refused", "an existence check", resp.status(), None);
            }
            None => self.note_unanswered("an existence check"),
        }
        false
    }

    /// Delete one object; returns whether the store
    /// accepted the deletion.
    async fn delete_object(&self, object: &str) -> bool {
        self.delete_object_if_match(object, None).await
    }

    /// Delete one object, conditionally on its ETag when one
    /// is given (`If-Match`): the store refuses with `412`
    /// when the object changed since it was read, and the
    /// delete then reports false. Returns whether the store
    /// accepted the deletion.
    async fn delete_object_if_match(&self, object: &str, etag: Option<&str>) -> bool {
        self.delete_answer(object, etag)
            .await
            .is_some_and(|status| status.is_success())
    }

    /// The store's answer to [`S3Store::delete_object_if_match`]'s
    /// delete: its status, or `None` when the request got no
    /// answer.
    async fn delete_answer(&self, object: &str, etag: Option<&str>) -> Option<reqwest::StatusCode> {
        let headers: Vec<(&str, &str)> = match etag {
            Some(etag) => vec![("if-match", etag)],
            None => Vec::new(),
        };
        let status = self
            .request_h("DELETE", object, &[], Vec::new(), &headers)
            .await?
            .status();
        // A store without conditional deletes may refuse the
        // precondition outright instead of ignoring it. The
        // caller has just read this copy and found it purged,
        // so delete it as before preconditions were sent;
        // otherwise no purge could remove anything from such
        // a store.
        if etag.is_some() && status == reqwest::StatusCode::NOT_IMPLEMENTED {
            return Some(
                self.request("DELETE", object, &[], Vec::new())
                    .await?
                    .status(),
            );
        }
        Some(status)
    }

    /// Purge one object if it holds a copy `target` covers
    /// that was written before `cutoff` (issue #446): read
    /// its header, then delete it conditionally on the ETag
    /// that read saw. A copy written after the purge began,
    /// or one that changed between the read and the delete,
    /// stays and is not counted. A read or a delete the
    /// store did not answer leaves the object unjudged
    /// (issue #462).
    ///
    /// Only the object this store writes the stored key to
    /// can be deleted (issue #608). Every entry this store
    /// wrote is there. Another deployment's entry, or an
    /// entry copied to another name, is kept whatever its
    /// header claims, so a header read without its MAC cannot
    /// steer a purge to it. `get` never serves those objects
    /// either: it reads only the name of the key it asks for.
    async fn purge_object(
        &self,
        object: &str,
        target: &PurgeTarget,
        cutoff: PurgeCutoff,
    ) -> ObjectPurge {
        let probed = match self.read_entry_header(object).await {
            HeaderRead::Entry(probed) => probed,
            HeaderRead::Nothing => return ObjectPurge::Kept,
            HeaderRead::Failed => return ObjectPurge::Failed,
            HeaderRead::Refused(status, code) => {
                self.note_refused(
                    Refused::Purge,
                    "a purge walk's read",
                    status,
                    code.as_deref(),
                );
                return ObjectPurge::Refused;
            }
        };
        let header = &probed.header;
        if object != self.object_key(&key_stem(&header.key)) {
            return ObjectPurge::Kept;
        }
        if !target.covers(&header.key, &header.groups) || !cutoff.covers(header.written) {
            return ObjectPurge::Kept;
        }
        match self.delete_answer(object, probed.etag.as_deref()).await {
            Some(status) if status.is_success() => ObjectPurge::Deleted,
            // Gone already, or replaced since it was read: no
            // copy the purge covers is left there.
            Some(reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::PRECONDITION_FAILED) => {
                ObjectPurge::Kept
            }
            Some(status) if store_failed(status) => ObjectPurge::Failed,
            // Any other answer is a refusal, such as the `403`
            // of credentials without `s3:DeleteObject` (issue
            // #709): the covered copy is still there, so the
            // walk is not complete and its tombstone stays.
            Some(status) => {
                self.note_refused(
                    Refused::Purge,
                    "a purge walk's delete of a covered copy",
                    status,
                    None,
                );
                ObjectPurge::Refused
            }
            None => ObjectPurge::Failed,
        }
    }

    /// Remove every entry whose stored key is `base` or one
    /// of its `#`-suffixed variants. Returns how many
    /// entries were removed.
    ///
    /// `base` is the purge target's canonical form;
    /// `epoch_base` is the form entries are actually keyed
    /// under in the current configuration (the base plus
    /// the config fingerprint segment, audit R2-08). The
    /// enumerable variants of the epoch base (itself and
    /// the four negotiated-class keys) are deleted directly
    /// by their deterministic object names, with no listing
    /// and no reads (audit R2-24); a header walk then
    /// catches any remaining spellings (request-hint
    /// variants, older epochs) by their stored keys.
    ///
    /// This is the unbounded form: the request path uses
    /// the bounded variant (issue #368), and this stays
    /// for the tests and any caller that wants the whole
    /// walk in one call. The same applies to the
    /// unbounded `purge_group` and `purge_source` below.
    pub async fn purge_matching(&self, base: &str, epoch_base: &str) -> usize {
        let target = PurgeTarget::Key(base.to_string());
        self.purge_whole(&target, Some(epoch_base), PurgeCutoff::EVERYTHING)
            .await
            .removed
    }

    /// Remove the copies `target` covers that were written
    /// before `cutoff` (issues #368, #446), with a bounded
    /// inline scan: the walk lists at most `budget` names,
    /// a page of them at a time (issue #709), and reads at
    /// most as many entry headers before returning, and any
    /// remainder comes back as a [`PendingPurge`] for
    /// [`S3Store::finish_purge`]: where the listing stands,
    /// never the rest of it. For a per-URL purge,
    /// `epoch_base` names the current epoch's form of the
    /// key, whose enumerable variants are reached directly
    /// by object name first; those reads are bounded by
    /// construction and always run inline. Whether the scan
    /// was complete covers the listing and the walk: the
    /// walk lists the variants and reads them again, so a
    /// failed read by name is one the walk repairs.
    pub async fn purge_bounded(
        &self,
        target: &PurgeTarget,
        epoch_base: Option<&str>,
        cutoff: PurgeCutoff,
        budget: usize,
    ) -> (Swept, Option<PendingPurge>) {
        let mut removed = 0;
        if let Some(epoch_base) = epoch_base {
            removed += self
                .purge_epoch_variants(epoch_base, target, cutoff)
                .await
                .removed;
        }
        let mut pass = Pass::default();
        let mut left = budget;
        let mut token: Option<String> = None;
        let rest = loop {
            // A page of no more names than the budget has left,
            // so the request lists no more than it reads.
            let max_keys = (left < LIST_PAGE_KEYS).then_some(left);
            let Some(page) = self.list_entry_page(token.as_deref(), max_keys).await else {
                pass.listing_failed = true;
                break None;
            };
            let EntryPage {
                mut objects,
                listed,
                next,
            } = page;
            // A store that lists more than it was asked for
            // leaves the excess, at most one page, to the rest.
            let unread = if objects.len() > left {
                objects.split_off(left)
            } else {
                Vec::new()
            };
            left = left.saturating_sub(listed.max(objects.len()));
            self.purge_objects(objects, target, cutoff, Some(&self.inline_reads), &mut pass)
                .await;
            match next {
                next if !unread.is_empty() => break Some((unread, next)),
                None => break None,
                Some(next) if left == 0 => break Some((Vec::new(), Some(next))),
                Some(next) => token = Some(next),
            }
        };
        self.note_pass(&pass);
        let swept = Swept {
            removed: removed + pass.removed,
            complete: pass.complete(),
        };
        let pending = rest.map(|(objects, next)| PendingPurge {
            target: target.clone(),
            cutoff,
            objects,
            next,
        });
        (swept, pending)
    }

    /// [`S3Store::purge_bounded`] with no budget, its remainder
    /// walked in the same call: the unbounded form the tests
    /// and the `purge_*` helpers use.
    async fn purge_whole(
        &self,
        target: &PurgeTarget,
        epoch_base: Option<&str>,
        cutoff: PurgeCutoff,
    ) -> Swept {
        let (inline, pending) = self
            .purge_bounded(target, epoch_base, cutoff, usize::MAX)
            .await;
        let Some(pending) = pending else {
            return inline;
        };
        let rest = self.finish_purge(pending).await;
        Swept {
            removed: inline.removed + rest.removed,
            complete: inline.complete && rest.complete,
        }
    }

    /// Purge the current epoch's enumerable variants of a
    /// key directly by object name (audit R2-24), under the
    /// same stamp rule as the walk. Five reads, bounded by
    /// construction: every per-URL purge runs them inline,
    /// including one that joins a running walk (issue #462).
    pub async fn purge_epoch_variants(
        &self,
        epoch_base: &str,
        target: &PurgeTarget,
        cutoff: PurgeCutoff,
    ) -> Swept {
        let objects: Vec<String> = [
            epoch_base.to_string(),
            format!("{epoch_base}#avif"),
            format!("{epoch_base}#jxl"),
            format!("{epoch_base}#webp"),
            format!("{epoch_base}#source"),
        ]
        .iter()
        .map(|candidate| self.object_key(&key_stem(candidate)))
        .collect();
        let mut pass = Pass::default();
        self.purge_objects(objects, target, cutoff, Some(&self.inline_reads), &mut pass)
            .await;
        self.note_pass(&pass);
        pass.swept()
    }

    /// Remove every entry belonging to `group`. Returns
    /// how many entries were removed.
    pub async fn purge_group(&self, group: &str) -> usize {
        let target = PurgeTarget::Group(group.to_string());
        self.purge_whole(&target, None, PurgeCutoff::EVERYTHING)
            .await
            .removed
    }

    /// Remove every entry whose result derives from
    /// `source`, whatever operations produced it (audit
    /// R2-07): an upload overwrite calls this so the
    /// replaced original's derivatives stop serving.
    /// Returns how many entries were removed.
    pub async fn purge_source(&self, source: &str) -> usize {
        let target = PurgeTarget::Source(source.to_string());
        self.purge_whole(&target, None, PurgeCutoff::EVERYTHING)
            .await
            .removed
    }

    /// Complete a bounded purge's remainder (issue #368):
    /// judge the names the inline budget left, then list on
    /// from its continuation token a page at a time (issue
    /// #709), deleting the covered copies. Returns how many
    /// entries were removed, and whether every object was
    /// judged. A background walk: the caller holds a
    /// [`S3Store::walk_turn`].
    pub async fn finish_purge(&self, pending: PendingPurge) -> Swept {
        let PendingPurge {
            target,
            cutoff,
            objects,
            next,
        } = pending;
        let from = next.map_or(PageFrom::Done, PageFrom::Token);
        let mut pass = Pass::default();
        self.walk_to_end(&target, cutoff, objects, from, &mut pass)
            .await;
        self.note_pass(&pass);
        pass.swept()
    }

    /// One whole pass over the bucket for the purges that
    /// joined a walk (issue #462): list it afresh, a page at a
    /// time, and purge what `target` covers under `cutoff`. A
    /// background walk: the caller holds a
    /// [`S3Store::walk_turn`].
    pub async fn relist(&self, target: &PurgeTarget, cutoff: PurgeCutoff) -> Swept {
        let mut pass = Pass::default();
        self.walk_to_end(target, cutoff, Vec::new(), PageFrom::First, &mut pass)
            .await;
        self.note_pass(&pass);
        pass.swept()
    }

    /// Judge `objects`, then every page of the listing from
    /// `from` to its end, one page at a time (issue #709): the
    /// walk holds one page of names, whatever the bucket
    /// holds. A listing page the store fails ends the walk
    /// incomplete. Reads are header-sized (see
    /// [`S3Store::read_entry_header`]), never whole entries.
    async fn walk_to_end(
        &self,
        target: &PurgeTarget,
        cutoff: PurgeCutoff,
        objects: Vec<String>,
        mut from: PageFrom,
        pass: &mut Pass,
    ) {
        self.purge_objects(objects, target, cutoff, None, pass)
            .await;
        loop {
            let token = match std::mem::replace(&mut from, PageFrom::Done) {
                PageFrom::Done => return,
                PageFrom::First => None,
                PageFrom::Token(token) => Some(token),
            };
            let Some(page) = self.list_entry_page(token.as_deref(), None).await else {
                pass.listing_failed = true;
                return;
            };
            self.purge_objects(page.objects, target, cutoff, None, pass)
                .await;
            if let Some(next) = page.next {
                from = PageFrom::Token(next);
            }
        }
    }

    /// Purge each of `objects` (see
    /// [`S3Store::purge_object`]), [`PURGE_READS_AT_ONCE`]
    /// at a time (issue #462): a header read is one round
    /// trip, and one after another they made a walk's time
    /// the object count times the store's latency. With
    /// `shared`, each read also holds one of its permits,
    /// a budget concurrent purges share (see
    /// [`PURGE_INLINE_READS_AT_ONCE`]). Adds to `pass` how
    /// many covered copies were deleted, and how many objects
    /// the store did not answer for or refused: the caller
    /// logs them once per pass (see [`S3Store::note_pass`]).
    async fn purge_objects(
        &self,
        objects: Vec<String>,
        target: &PurgeTarget,
        cutoff: PurgeCutoff,
        shared: Option<&tokio::sync::Semaphore>,
        pass: &mut Pass,
    ) {
        use futures_util::StreamExt as _;
        pass.judged += objects.len();
        // Each name moves into its read: a stream of borrowed
        // names makes the spawned walk fail the compiler's
        // Send check (a known limitation with higher-ranked
        // lifetimes).
        let (removed, unanswered, refused) = futures_util::stream::iter(objects)
            .map(move |object| async move {
                let _read = match shared {
                    Some(budget) => Some(
                        budget
                            .acquire()
                            .await
                            .expect("the purge read budget is never closed"),
                    ),
                    None => None,
                };
                self.purge_object(&object, target, cutoff).await
            })
            .buffer_unordered(PURGE_READS_AT_ONCE)
            .fold((0, 0, 0), |(removed, unanswered, refused), outcome| {
                std::future::ready(match outcome {
                    ObjectPurge::Deleted => (removed + 1, unanswered, refused),
                    ObjectPurge::Kept => (removed, unanswered, refused),
                    ObjectPurge::Failed => (removed, unanswered + 1, refused),
                    ObjectPurge::Refused => (removed, unanswered, refused + 1),
                })
            })
            .await;
        pass.removed += removed;
        pass.unanswered += unanswered;
        pass.refused += refused;
    }

    /// Count and log what a purge pass could not do, one line
    /// for its listing and one for its objects, as before the
    /// walk went page by page: a listing the store failed, and
    /// the objects it did not answer for or refused (each
    /// refusal also has its own count and line, see
    /// [`S3Store::note_refused`]).
    fn note_pass(&self, pass: &Pass) {
        if pass.listing_failed {
            self.note_error("purge walk", "listing the bucket failed");
        }
        let total = pass.judged;
        match (pass.unanswered, pass.refused) {
            (0, 0) => {}
            (failed, 0) => self.note_error(
                "purge walk",
                &format!("the store did not answer for {failed} of {total} objects"),
            ),
            (failed, refused) => self.note_error(
                "purge walk",
                &format!(
                    "the store did not answer for {failed} and refused {refused} of {total} objects"
                ),
            ),
        }
    }

    /// Lead or join the walk of `purge`'s target (issue
    /// #462). With no walk of an equal target running in
    /// this process, the caller leads one: the purge comes
    /// back with a [`WalkLease`] that marks the target as
    /// walking until the walk and its re-lists are done.
    /// Otherwise the purge joins the running walk and `None`
    /// comes back: the walk's leader takes the purge over,
    /// walks the bucket once more for it after its current
    /// pass, and ends it then.
    ///
    /// The extra pass is what keeps a joined purge whole.
    /// The running pass judges copies by the earlier purge's
    /// start, so it spares a copy rendered between the two
    /// purges, and its listing may predate that copy
    /// altogether; the joined purge's tombstone hides such a
    /// copy only until it expires, a grace period after the
    /// walk ends, so without a pass of its own the copy
    /// would serve again then.
    pub fn lead_or_join(self: &Arc<Self>, purge: SharedPurge) -> Option<(WalkLease, SharedPurge)> {
        let target = purge.active.tombstone().target.clone();
        let mut walks = self.walks.lock().unwrap();
        if let Some(joined) = walks.get_mut(&target) {
            joined.push(purge);
            return None;
        }
        walks.insert(target.clone(), Vec::new());
        drop(walks);
        let lease = WalkLease {
            store: Arc::clone(self),
            target,
            done: false,
        };
        Some((lease, purge))
    }

    /// Wait for a background walk's turn (issue #462): at
    /// most [`PURGE_WALKS_AT_ONCE`] walks hold one at a
    /// time, and the others queue in arrival order. The
    /// turn is returned when the permit is dropped.
    pub async fn walk_turn(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.walk_turns
            .acquire()
            .await
            .expect("the walk turn semaphore is never closed")
    }

    fn tombstone_prefix(&self) -> String {
        format!("{}{TOMBSTONE_DIR}", self.prefix)
    }

    /// Record `tombstone` in the bucket (issue #461), under
    /// a name unique to its purge: its start, its writer,
    /// and `sequence`, unique within the writer. Returns the
    /// object name, for marking the walk's end; `None` when
    /// the write failed, which is counted and logged: the
    /// purge then holds on this process only, as before
    /// tombstones existed.
    pub async fn put_tombstone(&self, tombstone: &Tombstone, sequence: u64) -> Option<String> {
        let object = format!(
            "{}{}-{:016x}-{sequence}",
            self.tombstone_prefix(),
            tombstone.cutoff.start_ms,
            tombstone.cutoff.writer
        );
        self.write_tombstone(&object, tombstone)
            .await
            .then_some(object)
    }

    /// Write `tombstone` to `object`: the first record, or
    /// the rewrite that marks its walk's end. Returns
    /// whether the store accepted it.
    pub async fn write_tombstone(&self, object: &str, tombstone: &Tombstone) -> bool {
        let data = serialize_tombstone(tombstone, &self.entry_key);
        if data.len() > MAX_TOMBSTONE_BYTES {
            self.note_error("cannot record a purge tombstone", "target too long");
            return false;
        }
        match self.request("PUT", object, &[], data).await {
            Some(resp) if resp.status().is_success() => true,
            Some(resp) => {
                self.note_error(
                    "cannot record a purge tombstone",
                    &format!("status {}", resp.status()),
                );
                false
            }
            None => {
                self.note_error("cannot record a purge tombstone", "transport error");
                false
            }
        }
    }

    /// Keep every tombstone of `target` from being marked ended
    /// while `disk`, one purge's disk deletions, are pending
    /// (issue #572). A restart before they are done indexes
    /// those files again, and only an unended tombstone keeps
    /// them misses. A later purge of the target has no
    /// deletions of its own for them, and its tombstone, once
    /// ended, supersedes the earlier ones, so it waits for them
    /// too. Called as the purge queues them, before it awaits
    /// anything: a later purge of the target can end first.
    pub fn hold_for_deletions(&self, target: &PurgeTarget, disk: &crate::storage::DeletionWait) {
        if disk.pending() {
            self.held_ends
                .lock()
                .unwrap()
                .entry(target.clone())
                .or_default()
                .deletions
                .push(disk.clone());
        }
    }

    /// Mark `tombstone`, stored at `object`, ended (issue
    /// #461): now when no purge of its target has disk
    /// deletions pending in this process, or else on the
    /// [`S3Store::end_held_purges`] after they are done. A
    /// held tombstone stays in force unended and covers every
    /// copy the target's earlier held ones do, so those are
    /// deleted from the bucket: repeated purges of a target
    /// whose deletion keeps failing would otherwise leave one
    /// unended tombstone each, none superseding another, and
    /// push older ones out of what a refresh reads.
    pub async fn end_tombstone(&self, object: &str, tombstone: &Tombstone) {
        let mut spent = Vec::new();
        let ready = {
            let mut held = self.held_ends.lock().unwrap();
            let entry = (object.to_string(), tombstone.clone());
            match held.get_mut(&tombstone.target) {
                None => vec![entry],
                Some(waiting) => {
                    waiting
                        .deletions
                        .retain(crate::storage::DeletionWait::pending);
                    if waiting.deletions.is_empty() {
                        let mut ready = held
                            .remove(&tombstone.target)
                            .map(|waiting| waiting.tombstones)
                            .unwrap_or_default();
                        ready.push(entry);
                        ready
                    } else {
                        let ended = Tombstone {
                            ended_ms: Some(0),
                            ..tombstone.clone()
                        };
                        waiting.tombstones.retain(|(earlier, held)| {
                            let superseded = ended.supersedes(held);
                            if superseded {
                                spent.push(earlier.clone());
                            }
                            !superseded
                        });
                        waiting.tombstones.push(entry);
                        Vec::new()
                    }
                }
            }
        };
        for object in &spent {
            self.delete_object(object).await;
        }
        self.write_ended(ready).await;
    }

    /// Mark ended the held tombstones of every target whose
    /// purges have all their disk files deleted now (issue
    /// #572), however long that took: run on every tombstone
    /// refresh. Ends within [`TOMBSTONE_REFRESH_DEADLINE`]; a
    /// tombstone whose write fails or does not finish by then
    /// stays in force, as when a walk's end cannot be written.
    pub async fn end_held_purges(&self) {
        let ready: Vec<(String, Tombstone)> = {
            let mut held = self.held_ends.lock().unwrap();
            let mut ready = Vec::new();
            held.retain(|_, waiting| {
                waiting
                    .deletions
                    .retain(crate::storage::DeletionWait::pending);
                if waiting.deletions.is_empty() {
                    ready.append(&mut waiting.tombstones);
                    return false;
                }
                true
            });
            ready
        };
        self.write_ended(ready).await;
    }

    /// Write each tombstone marked ended now, at most
    /// [`TOMBSTONE_READS_AT_ONCE`] at a time and within
    /// [`TOMBSTONE_REFRESH_DEADLINE`].
    async fn write_ended(&self, tombstones: Vec<(String, Tombstone)>) {
        use futures_util::StreamExt as _;
        if tombstones.is_empty() {
            return;
        }
        let ended_ms = Some(now_ms());
        let _ = tokio::time::timeout(
            TOMBSTONE_REFRESH_DEADLINE,
            futures_util::stream::iter(tombstones).for_each_concurrent(
                TOMBSTONE_READS_AT_ONCE,
                |(object, tombstone)| async move {
                    let ended = Tombstone {
                        ended_ms,
                        ..tombstone
                    };
                    self.write_tombstone(&object, &ended).await;
                },
            ),
        )
        .await;
    }

    /// Read one tombstone object: capped, authenticated, and
    /// consistent with the name it is stored under. Anything
    /// else (a plant, another deployment's record, a copy
    /// moved to another name) is ignored, never trusted, and
    /// so is an object that is gone (another replica deleted
    /// it after the listing). A store that does not answer
    /// for it is a failure (issue #549).
    async fn read_tombstone(&self, object: &str, start_ms: u64, writer: u64) -> TombstoneRead {
        let started = Instant::now();
        let Some(resp) = self.request_object(object, &[]).await else {
            return TombstoneRead::Failed("transport error".into());
        };
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return TombstoneRead::Ignored;
        }
        if !status.is_success() {
            return TombstoneRead::Failed(format!("status {status}"));
        }
        let data = match read_capped_at(resp, MAX_TOMBSTONE_BYTES, started).await {
            Ok(data) => data,
            Err(BodyFailure::TooLarge) => return TombstoneRead::Ignored,
            Err(BodyFailure::Failed) => {
                return TombstoneRead::Failed("body broke off or stalled".into());
            }
        };
        match parse_tombstone(&data, &self.entry_key) {
            Some(tombstone)
                if tombstone.cutoff.start_ms == start_ms && tombstone.cutoff.writer == writer =>
            {
                TombstoneRead::Read(tombstone)
            }
            _ => TombstoneRead::Ignored,
        }
    }

    /// [`S3Store::refresh_tombstones_within`] with
    /// [`TOMBSTONE_REFRESH_DEADLINE`].
    pub async fn refresh_tombstones(&self, guard: &PurgeGuard) -> bool {
        self.refresh_tombstones_within(guard, TOMBSTONE_REFRESH_DEADLINE)
            .await
    }

    /// Re-read the bucket's tombstones into `guard` (issue
    /// #461): list the tombstone sub-prefix, fetch what is
    /// new or changed since the last refresh (newest first,
    /// at most `MAX_SHARED_TOMBSTONES`,
    /// [`TOMBSTONE_READS_AT_ONCE`] at a time), and replace
    /// the guard's shared set. Tombstones whose grace has
    /// passed, or that a finished walk of the same target
    /// superseded, are deleted from the bucket on the way:
    /// every replica collects them, so no single process has
    /// to outlive its grace period.
    ///
    /// The whole refresh ends within `limit` (issue #549):
    /// one deadline covers the listing, the reads and the
    /// deletes. A listing that fails or runs out of time
    /// keeps the previous set, so a flaky store cannot make
    /// a purge lapse. The reads that finish by the deadline
    /// apply, and their ETags spare the next refresh from
    /// reading them again, so a slow store that keeps
    /// answering catches up, newest first. A tombstone read
    /// before whose read fails or does not finish in time
    /// keeps the version read before, under the ETag it was
    /// read with, so the next refresh reads it again. One
    /// never read before stays out of the set, as it always
    /// has, and every refresh retries it (issue #577 keeps it
    /// so: it is counted in [`S3Store::tombstones_unread`],
    /// named in the log line, and docs/deployment.md says how
    /// to close the gap). Each read the
    /// store fails counts as one tier error, the reads left
    /// at the deadline count as one together, and a refresh
    /// that has either logs one line for them. A read that
    /// finds the object gone is no failure: another replica
    /// collected it. Deletes still running at the deadline
    /// are dropped, and a later refresh lists those
    /// tombstones again. Tombstones listed past the newest
    /// `MAX_SHARED_TOMBSTONES` count as never read (issue
    /// #572). Returns whether the refresh listed and read
    /// everything.
    ///
    /// The refresh runs in stages: [`Self::list_tombstones`],
    /// [`plan_tombstone_reads`], [`Self::read_tombstones_by`],
    /// [`settle_tombstone_reads`], [`take_spent_tombstones`],
    /// [`Self::note_unread_tombstones`],
    /// [`Self::put_tombstones_in_force`] and
    /// [`Self::collect_spent_tombstones`].
    pub async fn refresh_tombstones_within(&self, guard: &PurgeGuard, limit: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + limit;
        let Some(listing) = self.list_tombstones(deadline, limit).await else {
            return false;
        };
        let previous = self.tombstones.lock().unwrap().clone();
        let (mut fresh, to_read) = plan_tombstone_reads(listing.named, &previous);
        let done = self.read_tombstones_by(deadline, &to_read).await;
        let mut settled =
            settle_tombstone_reads(&to_read, done, &previous, &mut fresh, listing.past_cap);
        let spent = take_spent_tombstones(&mut fresh, now_ms());
        self.note_unread_tombstones(&mut settled, to_read.len(), listing.past, &fresh, limit);
        self.put_tombstones_in_force(guard, fresh, settled.unread.len(), listing.listed_ms);
        self.collect_spent_tombstones(deadline, spent).await;
        settled.failed + settled.unfinished == 0 && listing.past == 0
    }

    /// The first stage of [`Self::refresh_tombstones_within`]:
    /// list the tombstone sub-prefix by `deadline`, and order
    /// what it names newest first. `None` when the listing
    /// failed or ran out of time, which is noted as a tier
    /// error.
    async fn list_tombstones(
        &self,
        deadline: tokio::time::Instant,
        limit: Duration,
    ) -> Option<TombstoneListing> {
        let prefix = self.tombstone_prefix();
        let Ok((listed, complete)) =
            tokio::time::timeout_at(deadline, self.list_under(&prefix)).await
        else {
            self.note_error("cannot list purge tombstones", &not_done_within(limit));
            return None;
        };
        if !complete {
            self.note_error("cannot list purge tombstones", "listing failed");
            return None;
        }
        let listed_ms = now_ms();
        let mut named: Vec<ListedTombstone> = listed
            .into_iter()
            .filter_map(|(object, etag)| {
                let (start_ms, writer) = parse_tombstone_name(object.strip_prefix(&prefix)?)?;
                Some((start_ms, writer, object, etag))
            })
            .collect();
        named.sort_by_key(|(start_ms, ..)| std::cmp::Reverse(*start_ms));
        // The ones past the cap are not read and do not apply
        // (issue #572): they count as unread, and the line
        // `note_unread_tombstones` writes names them, so a
        // tombstone that newer ones pushed out does not lapse
        // unseen.
        let past_cap: Vec<(u64, String)> = named
            .split_off(MAX_SHARED_TOMBSTONES.min(named.len()))
            .into_iter()
            .map(|(start_ms, _, object, _)| (start_ms, object))
            .collect();
        Some(TombstoneListing {
            named,
            past: past_cap.len(),
            past_cap,
            listed_ms,
        })
    }

    /// Read `to_read` by `deadline`, [`TOMBSTONE_READS_AT_ONCE`]
    /// at a time. Reads finish in any order; the ones still out
    /// at the deadline are dropped with the stream, and the ones
    /// not yet started never start. Each read owns its name, as
    /// in `purge_objects`.
    async fn read_tombstones_by(
        &self,
        deadline: tokio::time::Instant,
        to_read: &[ListedTombstone],
    ) -> HashMap<String, TombstoneRead> {
        use futures_util::StreamExt as _;
        let mut done: HashMap<String, TombstoneRead> = HashMap::new();
        let _ = tokio::time::timeout_at(deadline, async {
            let mut reads = futures_util::stream::iter(to_read.to_vec())
                .map(|(start_ms, writer, object, _)| async move {
                    let read = self.read_tombstone(&object, start_ms, writer).await;
                    (object, read)
                })
                .buffer_unordered(TOMBSTONE_READS_AT_ONCE);
            while let Some((object, read)) = reads.next().await {
                done.insert(object, read);
            }
        })
        .await;
        done
    }

    /// Note the reads this refresh could not make, when there
    /// were any: `failed` counts the reads the store answered
    /// with a failure, one error each, as every other store
    /// request counts. The reads the deadline cut off or kept
    /// from starting count one error together: their number
    /// grows with the tombstones listed, not with the requests
    /// the store failed. The tombstones listed past the cap
    /// are no failure of the store, so no tier error: the
    /// unread count shows them.
    fn note_unread_tombstones(
        &self,
        settled: &mut SettledReads,
        total: usize,
        past: usize,
        fresh: &HashMap<String, (String, Tombstone)>,
        limit: Duration,
    ) {
        let SettledReads {
            failed,
            unfinished,
            kept,
            unread,
            first_failure,
        } = settled;
        let (failed, unfinished) = (*failed, *unfinished);
        if failed + unfinished == 0 && past == 0 {
            return;
        }
        // A kept version that turned out spent was dropped
        // from the set and is collected at the end of the
        // refresh, so only the kept versions still in force
        // count as kept.
        let kept_in_force = kept.iter().filter(|o| fresh.contains_key(*o)).count();
        let mut what = Vec::new();
        if let Some(why) = first_failure {
            what.push(format!("{failed} of {total} reads failed (first: {why})"));
        }
        if unfinished > 0 {
            what.push(format!(
                "{unfinished} of {total} {}",
                not_done_within(limit)
            ));
        }
        if past > 0 {
            what.push(format!(
                "{past} listed past the newest {MAX_SHARED_TOMBSTONES} not read"
            ));
        }
        // The names come last, so a match on the text
        // before them still holds.
        let mut detail = format!(
            "{}; kept as last read: {kept_in_force}; not in force until read: {}",
            what.join(", "),
            unread.len()
        );
        if !unread.is_empty() {
            detail += &format!("; oldest not in force: {}", oldest_unread(unread));
        }
        self.note_errors(
            failed + u64::from(unfinished > 0),
            "cannot read purge tombstones",
            &detail,
        );
    }

    /// Put the refreshed set in force: in `guard`, as the set
    /// the next refresh compares against, and in the counts
    /// `/metrics` reports, stored once the guard holds every
    /// tombstone this listing read (issue #577).
    fn put_tombstones_in_force(
        &self,
        guard: &PurgeGuard,
        fresh: HashMap<String, (String, Tombstone)>,
        unread: usize,
        listed_ms: u64,
    ) {
        guard.replace_shared(
            fresh
                .iter()
                .map(|(object, (_, tombstone))| (object.clone(), tombstone.clone()))
                .collect(),
        );
        *self.tombstones.lock().unwrap() = fresh;
        self.tombstones_unread
            .store(unread as u64, std::sync::atomic::Ordering::Relaxed);
        self.tombstones_listed_ms
            .store(listed_ms, std::sync::atomic::Ordering::Relaxed);
    }

    /// Delete the spent tombstones from the bucket by
    /// `deadline`. Collection can wait: a spent tombstone left
    /// in the bucket at the deadline is listed, read and
    /// deleted by a later refresh.
    async fn collect_spent_tombstones(&self, deadline: tokio::time::Instant, spent: Vec<String>) {
        use futures_util::StreamExt as _;
        let _ = tokio::time::timeout_at(
            deadline,
            futures_util::stream::iter(spent).for_each_concurrent(
                TOMBSTONE_READS_AT_ONCE,
                |object| async move {
                    self.delete_object(&object).await;
                },
            ),
        )
        .await;
    }
}

/// A listed tombstone: the start and writer its name gives,
/// its object name, and its ETag.
type ListedTombstone = (u64, u64, String, String);

/// What a refresh's listing named (see
/// [`S3Store::list_tombstones`]).
struct TombstoneListing {
    /// The newest [`MAX_SHARED_TOMBSTONES`], newest first.
    named: Vec<ListedTombstone>,
    /// The ones listed past them: the start each name gives,
    /// and the name (issue #572).
    past_cap: Vec<(u64, String)>,
    /// How many of them there are.
    past: usize,
    /// When the listing completed, in unix milliseconds.
    listed_ms: u64,
}

/// How a refresh's reads ended (see
/// [`settle_tombstone_reads`]).
struct SettledReads {
    /// The reads the store answered with a failure.
    failed: u64,
    /// The reads the deadline cut off or kept from starting.
    unfinished: u64,
    /// The tombstones whose version read before stays in
    /// force because their read failed or did not finish.
    kept: Vec<String>,
    /// The tombstones never read, with the start their names
    /// give (issue #577): the ones listed past the cap, and
    /// the ones read for the first time that failed or did
    /// not finish.
    unread: Vec<(u64, String)>,
    /// Why the first failed read failed.
    first_failure: Option<String>,
}

/// The listed tombstones split by whether they need reading:
/// one listed under the ETag the last refresh read it with
/// stands as read, in the set returned; the rest are to read.
fn plan_tombstone_reads(
    named: Vec<ListedTombstone>,
    previous: &HashMap<String, (String, Tombstone)>,
) -> (HashMap<String, (String, Tombstone)>, Vec<ListedTombstone>) {
    let mut fresh: HashMap<String, (String, Tombstone)> = HashMap::new();
    let mut to_read = Vec::new();
    for (start_ms, writer, object, etag) in named {
        if !etag.is_empty()
            && let Some((seen, tombstone)) = previous.get(&object)
            && *seen == etag
        {
            fresh.insert(object, (etag, tombstone.clone()));
            continue;
        }
        to_read.push((start_ms, writer, object, etag));
    }
    (fresh, to_read)
}

/// Apply the reads that finished to `fresh`, and settle the
/// rest: an authentic tombstone joins the set under its new
/// ETag, an ignored one stays out, and one whose read failed
/// or did not finish keeps the version read before, under
/// its own ETag, so the next refresh reads the object again.
/// One never read before is unread, with the ones listed past
/// the cap (`past_cap`).
fn settle_tombstone_reads(
    to_read: &[ListedTombstone],
    mut done: HashMap<String, TombstoneRead>,
    previous: &HashMap<String, (String, Tombstone)>,
    fresh: &mut HashMap<String, (String, Tombstone)>,
    past_cap: Vec<(u64, String)>,
) -> SettledReads {
    let mut settled = SettledReads {
        failed: 0,
        unfinished: 0,
        kept: Vec::new(),
        unread: past_cap,
        first_failure: None,
    };
    for (start_ms, _, object, etag) in to_read {
        match done.remove(object) {
            Some(TombstoneRead::Read(tombstone)) => {
                fresh.insert(object.clone(), (etag.clone(), tombstone));
                continue;
            }
            Some(TombstoneRead::Ignored) => continue,
            Some(TombstoneRead::Failed(why)) => {
                settled.failed += 1;
                settled.first_failure.get_or_insert(why);
            }
            None => settled.unfinished += 1,
        }
        if let Some(known) = previous.get(object) {
            fresh.insert(object.clone(), known.clone());
            settled.kept.push(object.clone());
        } else {
            settled.unread.push((*start_ms, object.clone()));
        }
    }
    settled
}

/// Take out of `fresh` the tombstones that are spent at
/// `now`: those whose grace has passed, and those a newer one
/// in the set supersedes. Returns their names, for
/// collection.
fn take_spent_tombstones(
    fresh: &mut HashMap<String, (String, Tombstone)>,
    now: u64,
) -> Vec<String> {
    let spent: Vec<String> = fresh
        .iter()
        .filter(|(object, (_, tombstone))| {
            !tombstone.live(now)
                || fresh
                    .iter()
                    .any(|(other, (_, newer))| other != *object && newer.supersedes(tombstone))
        })
        .map(|(object, _)| object.clone())
        .collect();
    for object in &spent {
        fresh.remove(object);
    }
    spent
}

/// How a refresh that ran out of time words it.
fn not_done_within(limit: Duration) -> String {
    format!("not done within {}s", limit.as_secs_f32())
}

/// What reading one listed tombstone found (issue #549).
enum TombstoneRead {
    /// An authentic tombstone, consistent with its name.
    Read(Tombstone),
    /// Nothing to apply: the object is gone, or is not a
    /// tombstone this deployment wrote.
    Ignored,
    /// The store did not answer for it, for the reason
    /// given.
    Failed(String),
}

/// The un-scanned remainder of a bounded purge walk
/// (issue #368), with the rule that selects among it: where
/// the listing stands, not the rest of the listing (issue
/// #709). A walk waiting for its turn keeps the continuation
/// token of the next page, and, only when the store listed
/// more than the inline scan asked for, the names of that
/// one page it did not judge; before, it kept every name in
/// the bucket, about 3.5 MB a purge at 31,000 objects. Hand
/// it to [`S3Store::finish_purge`] to complete the purge off
/// the request path.
pub struct PendingPurge {
    target: PurgeTarget,
    cutoff: PurgeCutoff,
    /// Entry objects listed but not judged yet: at most one
    /// page's.
    objects: Vec<String>,
    /// The continuation token of the page after them; `None`
    /// when theirs was the last.
    next: Option<String>,
}

impl PendingPurge {
    /// How many listed names the remainder holds: at most one
    /// page's, whatever the bucket holds.
    #[cfg(test)]
    fn held_names(&self) -> usize {
        self.objects.len()
    }
}

/// One page of a ListObjectsV2 listing.
struct ListPage {
    /// Each listed object's key and ETag, in listing order.
    entries: Vec<(String, String)>,
    /// The continuation token of the next page; `None` when
    /// this page is the last.
    next: Option<String>,
}

/// One listing page's entry objects (see
/// [`S3Store::list_entry_page`]).
struct EntryPage {
    /// The names of the entry form, in listing order.
    objects: Vec<String>,
    /// How many names the page listed, entries or not.
    listed: usize,
    next: Option<String>,
}

/// Where a walk's listing goes on (issue #709).
enum PageFrom {
    /// From the first page.
    First,
    /// From the page a continuation token names.
    Token(String),
    /// Nowhere: the last page has been listed.
    Done,
}

/// What one purge pass did, page by page (issue #709): the
/// copies it deleted, the objects it judged, the ones the
/// store did not answer for and the ones it refused, and
/// whether a listing page failed.
#[derive(Debug, Default)]
struct Pass {
    removed: usize,
    judged: usize,
    unanswered: usize,
    refused: usize,
    listing_failed: bool,
}

impl Pass {
    /// Whether the pass judged every object in the bucket: its
    /// listing ran to the end, and every object was answered.
    fn complete(&self) -> bool {
        !self.listing_failed && self.unanswered == 0 && self.refused == 0
    }

    fn swept(&self) -> Swept {
        Swept {
            removed: self.removed,
            complete: self.complete(),
        }
    }
}

/// What a purge pass did (issue #462): how many covered
/// copies it deleted, and whether it judged every object in
/// the bucket. A pass is incomplete when its listing
/// stopped early, or when the store did not answer, or
/// refused (issue #709), a header read or a covered copy's
/// delete. Such a pass must not mark its purges ended: a
/// copy it missed would serve again once their tombstones
/// expire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Swept {
    pub removed: usize,
    pub complete: bool,
}

/// What a purge did to one object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectPurge {
    /// It held a covered copy, now deleted.
    Deleted,
    /// It holds nothing the purge removes: no entry, an
    /// entry the purge does not cover or that was written
    /// after the purge began, or a copy replaced since it
    /// was read.
    Kept,
    /// The store did not answer the read or the delete, so
    /// the object was not judged.
    Failed,
    /// The store refused the read or the delete (issue #709),
    /// a `403` say: the object was not judged, or its covered
    /// copy is still there.
    Refused,
}

/// The outcome of reading one object's header for a purge.
enum HeaderRead {
    /// The object holds an entry.
    Entry(ProbedEntry),
    /// Nothing to purge: the object is gone, is not a cache
    /// entry, or is past the entry size ceiling.
    Nothing,
    /// The store did not answer (see [`store_failed`]).
    Failed,
    /// The store refused the read with this status (issue
    /// #709), and the error code its answer named, if any: an
    /// `AccessDenied` for the credentials, an
    /// `InvalidObjectState` for an object in an archive
    /// storage class.
    Refused(reqwest::StatusCode, Option<String>),
}

impl HeaderRead {
    /// A read the store answered with an error `status`: a
    /// `404` (gone since the listing) or a `416` (an empty
    /// object, which a range cannot cover) holds nothing to
    /// purge. Before issue #709 every other error that was not
    /// a store failure read as nothing to purge too, so a walk
    /// whose reads the store refused (a `403`) ended its purge
    /// unjudged.
    fn unanswered(status: reqwest::StatusCode) -> Self {
        if store_failed(status) {
            HeaderRead::Failed
        } else if matches!(
            status,
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::RANGE_NOT_SATISFIABLE
        ) {
            HeaderRead::Nothing
        } else {
            HeaderRead::Refused(status, None)
        }
    }

    /// [`HeaderRead::unanswered`] for the error answer `resp`,
    /// with a refusal's error code read from the front of its
    /// body (see [`refusal_code`]), so the line it logs says
    /// why.
    async fn of_error(resp: reqwest::Response, started: Instant) -> Self {
        match Self::unanswered(resp.status()) {
            HeaderRead::Refused(status, _) => {
                HeaderRead::Refused(status, refusal_code(resp, started).await)
            }
            other => other,
        }
    }
}

/// Whether an error status means the store did not act on
/// a request, rather than that the request had no object
/// to act on (issue #462): a server error, throttling, or a
/// timeout. A purge pass that meets one has not judged that
/// object; a 404 or a 412 is an answer, and any other error
/// is a refusal (issue #709), which leaves the object
/// unjudged too.
fn store_failed(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
}

/// A purge on its way through the shared tier (issues
/// #461, #462): the handle that keeps it running, its
/// tombstone's object name in the bucket (`None` when it
/// could not be recorded there), and the audit event and
/// target its walk's end is logged under.
#[derive(Debug)]
pub struct SharedPurge {
    pub active: ActivePurge,
    pub tombstone: Option<String>,
    /// The purge's disk deletions (issue #572): its tombstone
    /// is marked ended only once they are done.
    pub disk: Option<crate::storage::DeletionWait>,
    pub event: &'static str,
    pub log_target: String,
}

/// What keeps a target's tombstones from being marked ended
/// (issue #572): the disk deletions of its purges in this
/// process not known to be done, and the tombstones to mark
/// once they are.
#[derive(Debug, Default)]
struct HeldEnds {
    deletions: Vec<crate::storage::DeletionWait>,
    tombstones: Vec<(String, Tombstone)>,
}

/// A target's walk, held by the purge that leads it
/// (issue #462) until no identical purge is waiting on it.
/// Dropping the lease before then (the walk's task panicked
/// or was torn down) frees the target and drops the purges
/// that joined: each one's bucket tombstone, never marked
/// ended, stays in force like that of a walk whose process
/// stopped, until a later purge of the target finishes its
/// walk.
pub struct WalkLease {
    store: Arc<S3Store>,
    target: PurgeTarget,
    done: bool,
}

impl WalkLease {
    /// The purges that joined the walk since the last call.
    /// Empty means the walk is over: the target is freed in
    /// the same step, so an identical purge that arrives
    /// after it leads a walk of its own instead of joining
    /// one that has ended.
    pub fn take_joined(&mut self) -> Vec<SharedPurge> {
        let mut walks = self.store.walks.lock().unwrap();
        let joined = walks
            .get_mut(&self.target)
            .map(std::mem::take)
            .unwrap_or_default();
        if joined.is_empty() {
            walks.remove(&self.target);
            self.done = true;
        }
        joined
    }
}

impl Drop for WalkLease {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let joined = self
            .store
            .walks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.target);
        // Dropped with the registry unlocked: each purge
        // takes the guard's lock as it ends.
        drop(joined);
    }
}

/// The outcome of reading one entry's header for a purge:
/// what the header claims, and the object's ETag at the
/// moment it was read, for a conditional delete.
struct ProbedEntry {
    header: HeaderProbe,
    etag: Option<String>,
}

/// Why [`read_capped_at`] returned no body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyFailure {
    /// The body passed the cap: refused, not a store fault.
    TooLarge,
    /// The body broke off, or did not arrive whole within
    /// its budget (issue #550): a store fault.
    Failed,
}

/// Read a response body under `cap` bytes, streaming so an
/// oversized object is refused as soon as it passes the cap
/// instead of after it has been buffered whole. Both
/// failures read as a miss upstream; a lookup also counts a
/// [`BodyFailure::Failed`] as a tier error.
///
/// The read is bounded too (issue #550): it must end within
/// the [`transfer_budget`] of the length the store declared,
/// counted from `started`, when the request began, so a
/// small object keeps the short bound and a large one gets
/// time in proportion. A store that declares no length gets
/// the budget of what it has sent so far, so it must keep
/// up the minimum rate. The client's read timeout ends a
/// body that stalls outright sooner.
async fn read_capped_at(
    mut resp: reqwest::Response,
    cap: usize,
    started: Instant,
) -> Result<Vec<u8>, BodyFailure> {
    let declared = resp.content_length();
    if declared.is_some_and(|n| n > cap as u64) {
        return Err(BodyFailure::TooLarge);
    }
    let mut data = Vec::new();
    loop {
        let expected = declared.unwrap_or(0).max(data.len() as u64);
        let deadline = tokio::time::Instant::from_std(started + transfer_budget(expected));
        let Some(chunk) = tokio::time::timeout_at(deadline, resp.chunk())
            .await
            .map_err(|_| BodyFailure::Failed)?
            .map_err(|_| BodyFailure::Failed)?
        else {
            break;
        };
        if data
            .len()
            .checked_add(chunk.len())
            .is_none_or(|len| len > cap)
        {
            return Err(BodyFailure::TooLarge);
        }
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

/// The error code an error answer's body names (issue #709),
/// S3's `<Code>` (`AccessDenied`, `ExpiredToken`,
/// `SignatureDoesNotMatch`, `SlowDown`): only the first
/// [`REFUSAL_BODY_BYTES`] are read, where S3 puts it, and only a
/// short name of letters and digits is kept, so the line it
/// goes into says why without carrying what else the store
/// sent.
async fn refusal_code(resp: reqwest::Response, started: Instant) -> Option<String> {
    let body = read_prefix_at(resp, REFUSAL_BODY_BYTES, started).await?;
    let text = String::from_utf8_lossy(&body);
    let code = xml_field(&text, "Code")?;
    (!code.is_empty() && code.len() <= 64 && code.bytes().all(|b| b.is_ascii_alphanumeric()))
        .then_some(code)
}

/// The first `cap` bytes of a response body, or all of a
/// shorter one, under the bound of that many bytes from
/// `started` (issue #550); the rest is never read. `None` when
/// the body broke off or ran out of time first.
async fn read_prefix_at(
    mut resp: reqwest::Response,
    cap: usize,
    started: Instant,
) -> Option<Vec<u8>> {
    let deadline = tokio::time::Instant::from_std(started + transfer_budget(cap as u64));
    let mut data = Vec::new();
    while data.len() < cap {
        let Some(chunk) = tokio::time::timeout_at(deadline, resp.chunk())
            .await
            .ok()?
            .ok()?
        else {
            break;
        };
        let room = cap - data.len();
        data.extend_from_slice(&chunk[..chunk.len().min(room)]);
    }
    Some(data)
}

/// The ETag a store reported for an object, verbatim.
fn response_etag(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Parse a tombstone's object name below the sub-prefix,
/// `<start-ms>-<writer, 16 hex>-<sequence>`, into its start
/// and writer. Anything else is not a tombstone this build
/// wrote, and is never fetched.
fn parse_tombstone_name(name: &str) -> Option<(u64, u64)> {
    let mut parts = name.split('-');
    let (start, writer, sequence) = (parts.next()?, parts.next()?, parts.next()?);
    let digits = |s: &str| !s.is_empty() && s.len() <= 20 && s.bytes().all(|b| b.is_ascii_digit());
    if parts.next().is_some()
        || !digits(start)
        || !digits(sequence)
        || writer.len() != 16
        || !writer.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return None;
    }
    Some((start.parse().ok()?, u64::from_str_radix(writer, 16).ok()?))
}

/// The three of `unread`'s objects whose purges began first,
/// by the start their names give, for the `cannot read purge
/// tombstones` line (issue #577): "a, b, c and N more". The
/// names passed [`parse_tombstone_name`] (digits, 16 hex
/// digits, digits) under the configured prefix, so they are
/// safe to log.
fn oldest_unread(unread: &mut [(u64, String)]) -> String {
    const NAMED: usize = 3;
    unread.sort();
    let names: Vec<&str> = unread
        .iter()
        .take(NAMED)
        .map(|(_, object)| object.as_str())
        .collect();
    let mut line = names.join(", ");
    if unread.len() > NAMED {
        line += &format!(" and {} more", unread.len() - NAMED);
    }
    line
}

/// The unescaped text of the first `<tag>` element in
/// `xml`, if any.
fn xml_field(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml_unescape(&xml[start..end]))
}

/// The five predefined XML entities, the only escaping S3
/// applies to listed keys.
fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// The fake's instruments for pacing purge walks (issue
    /// #462).
    #[derive(Default)]
    pub struct WalkProbe {
        /// While set, a ranged GET of an object whose name
        /// starts with this waits until it is cleared, so a
        /// test can park walks on chosen objects while the
        /// others pass.
        pub hold_prefix: StdMutex<Option<String>>,
        /// Ranged GETs `hold_prefix` parks right now, and the
        /// most it has parked at once.
        pub held: AtomicUsize,
        pub held_peak: AtomicUsize,
        /// How long each ranged GET waits before it is
        /// answered: a stand-in for the store's round trip.
        pub delay_ms: AtomicU64,
        /// Ranged GETs waiting on that delay right now, and
        /// the most at once.
        pub delayed: AtomicUsize,
        pub delayed_peak: AtomicUsize,
        /// The prefix of every listing's first page served, in
        /// order: one per pass over the prefix.
        pub lists: StdMutex<Vec<String>>,
        /// Every listing page served, first pages included,
        /// with the prefix it listed and the most names it
        /// asked for (issue #709).
        pub pages: StdMutex<Vec<(String, Option<usize>)>>,
    }

    impl WalkProbe {
        /// How many listings of `prefix` itself were begun:
        /// the purge walks' passes over a store under that
        /// prefix (a tombstone refresh lists a sub-prefix).
        /// A pass lists its pages after the first from their
        /// continuation tokens, and those are not counted.
        pub fn lists_of(&self, prefix: &str) -> usize {
            self.lists
                .lock()
                .unwrap()
                .iter()
                .filter(|p| *p == prefix)
                .count()
        }

        /// How many listing pages of `prefix` were served.
        pub fn pages_of(&self, prefix: &str) -> usize {
            self.pages
                .lock()
                .unwrap()
                .iter()
                .filter(|(p, _)| p == prefix)
                .count()
        }
    }

    /// The fake's instrument for hanging requests by name
    /// (issue #549), so a test can hang the tombstone
    /// requests alone. While one of the prefixes is set, a
    /// request of its kind for a name that starts with it
    /// waits until it is cleared, then is handled against
    /// the store as it stands.
    #[derive(Default)]
    pub struct NameHold {
        /// Whole-object GETs, by object name.
        pub reads: StdMutex<Option<String>>,
        /// Listings, by the prefix they list.
        pub lists: StdMutex<Option<String>>,
        /// DELETEs, by object name.
        pub deletes: StdMutex<Option<String>>,
        /// Requests held right now, and the most at once.
        pub held: AtomicUsize,
        pub held_peak: AtomicUsize,
    }

    impl NameHold {
        fn holds(kind: &StdMutex<Option<String>>, name: &str) -> bool {
            kind.lock()
                .unwrap()
                .as_deref()
                .is_some_and(|prefix| name.starts_with(prefix))
        }

        async fn wait(&self, kind: &StdMutex<Option<String>>, name: &str) {
            if !Self::holds(kind, name) {
                return;
            }
            let now = self.held.fetch_add(1, Ordering::SeqCst) + 1;
            self.held_peak.fetch_max(now, Ordering::SeqCst);
            while Self::holds(kind, name) {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            self.held.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// The fake's instrument for breaking whole-object GETs
    /// by name (issue #549), the store failures that are not
    /// an answer: while one of the prefixes is set, a GET of
    /// a name that starts with it fails that way.
    #[derive(Default)]
    pub struct NameBreak {
        /// The connection closes before any answer.
        pub unanswered: StdMutex<Option<String>>,
        /// The answer is a 200 whose body breaks off: half
        /// of it is sent under a content-length for all of
        /// it, and the connection closes.
        pub cut: StdMutex<Option<String>>,
    }

    /// Injectable failure switches for the error-path
    /// tests: while a flag is set, the matching operation
    /// answers 500 instead of doing its work.
    #[derive(Default)]
    pub struct FailFlags {
        pub get_object: bool,
        pub put: bool,
        pub delete: bool,
        pub list: bool,
        /// Answer a conditional DELETE with 501, as a store
        /// without conditional deletes may.
        pub if_match_delete: bool,
        /// Refuse every DELETE with S3's `403 AccessDenied`,
        /// as a store refuses credentials without
        /// `s3:DeleteObject` (issue #709).
        pub refuse_delete: bool,
        /// Refuse every object GET, ranged or not, with
        /// `403 AccessDenied` (issue #709).
        pub refuse_get: bool,
        /// Answer every listing page as truncated, naming as
        /// the next page the one it was asked for: a store
        /// whose listing never ends (issue #709).
        pub stuck_token: bool,
        /// Answer every request for an object (a GET, ranged
        /// or not, a HEAD, a PUT, a DELETE) with S3's `307
        /// TemporaryRedirect` to the same path and query
        /// under this base (issue #781).
        pub redirect_objects: Option<String>,
        /// Answer every listing page the same way.
        pub redirect_listings: Option<String>,
    }

    /// The credentials the fake accepts until a test sets
    /// others (issue #709): every test store signs with them,
    /// and the fake checks every request's SigV4 signature
    /// against them, as S3 does, refusing a bad one in S3's
    /// error shape.
    pub fn test_credentials() -> crate::sigv4::Credentials {
        crate::sigv4::Credentials {
            access_key: "AKIDFAKES3TEST".to_string(),
            secret_key: "fake/s3+test/secret-key".to_string(),
            session_token: None,
        }
    }

    /// [`test_credentials`] as a store takes them.
    pub fn test_aws() -> crate::credentials::AwsCredentials {
        test_credentials().into()
    }

    /// The region the fake signs in, as every test store does.
    const FAKE_REGION: &str = "us-east-1";

    /// A minimal in-process S3: PUT/GET/DELETE plus
    /// ListObjectsV2 over one shared map, path-style URLs.
    pub struct FakeS3 {
        pub objects: Arc<StdMutex<HashMap<String, Vec<u8>>>>,
        pub fail: Arc<StdMutex<FailFlags>>,
        /// One line per request served, `"<METHOD> <key>"`
        /// (plus the conditional-write marker when the
        /// request carried one, and the content type when
        /// the request carried one), so tests can assert
        /// how a lookup was answered, not just what it
        /// returned.
        pub requests: Arc<StdMutex<Vec<String>>>,
        /// Ranged GETs (the purge walk's header reads)
        /// that have arrived so far, held ones included.
        pub ranged_gets: Arc<AtomicUsize>,
        /// While set to `Some(n)`, every ranged GET past
        /// the first `n` waits until it is cleared, so a
        /// test can act while a purge walk is mid-flight.
        pub hold_ranged_after: Arc<StdMutex<Option<usize>>>,
        /// While set, a whole-object GET (a lookup's read)
        /// is answered from the object as it stands when the
        /// request arrives, then held before the answer is
        /// sent until cleared, so a test can change the
        /// store while those bytes are in flight.
        pub hold_object_gets: Arc<AtomicBool>,
        /// Whole-object GETs held right now.
        pub held_object_gets: Arc<AtomicUsize>,
        /// While set, a DELETE waits before it is handled,
        /// so a test can replace the object between a purge
        /// walk's header read and its delete.
        pub hold_deletes: Arc<AtomicBool>,
        /// DELETEs held right now.
        pub held_deletes: Arc<AtomicUsize>,
        /// While set, a PUT waits before it is handled, so
        /// a test can keep write-throughs in flight (issue
        /// #463).
        pub hold_puts: Arc<AtomicBool>,
        /// PUTs held right now.
        pub held_puts: Arc<AtomicUsize>,
        /// Purge walk pacing instruments (issue #462).
        pub walk: Arc<WalkProbe>,
        /// Requests held by name (issue #549).
        pub name_hold: Arc<NameHold>,
        /// Whole-object GETs broken by name (issue #549).
        pub name_break: Arc<NameBreak>,
        /// The credentials every request must be signed with
        /// (issue #709): [`test_credentials`] to start with.
        /// `None` serves unsigned requests, as a store that
        /// allows anonymous access does.
        pub credentials: Arc<StdMutex<Option<crate::sigv4::Credentials>>>,
        /// One line per request the signature check refused,
        /// `"<METHOD> <key> <code>"`.
        pub refused: Arc<StdMutex<Vec<String>>>,
        pub endpoint: String,
    }

    impl FakeS3 {
        /// The requests the signature check refused so far.
        pub fn refused(&self) -> Vec<String> {
            self.refused.lock().unwrap().clone()
        }
    }

    /// The first 32 hex digits of a filler object that lists
    /// ahead of every entry object, and of one that lists
    /// after every entry object. Entry names are SHA-256
    /// digests, so no real one starts with either.
    pub const HEAD_FILLER: &str = "00000000000000000000000000000000";
    pub const TAIL_FILLER: &str = "ffffffffffffffffffffffffffffffff";

    /// Filler object `i` after `lead` (`HEAD_FILLER` or
    /// `TAIL_FILLER`). It has the form of an entry name,
    /// `<64 hex>.bin`, so a purge walk lists it and reads its
    /// header (issue #608), but it holds no entry, so the
    /// walk leaves it.
    pub fn filler_name(lead: &str, i: usize) -> String {
        format!("{lead}{i:032x}.bin")
    }

    /// The ETag the fake reports for an object's bytes:
    /// quoted, and different whenever the bytes differ.
    pub fn etag_of(data: &[u8]) -> String {
        use sha2::Digest as _;
        format!("\"{}\"", hex::encode(&sha2::Sha256::digest(data)[..16]))
    }

    pub async fn start_fake() -> FakeS3 {
        let objects = Arc::new(StdMutex::new(HashMap::new()));
        let fail = Arc::new(StdMutex::new(FailFlags::default()));
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let ranged_gets = Arc::new(AtomicUsize::new(0));
        let hold_ranged_after: Arc<StdMutex<Option<usize>>> = Arc::new(StdMutex::new(None));
        let hold_object_gets = Arc::new(AtomicBool::new(false));
        let held_object_gets = Arc::new(AtomicUsize::new(0));
        let hold_deletes = Arc::new(AtomicBool::new(false));
        let held_deletes = Arc::new(AtomicUsize::new(0));
        let hold_puts = Arc::new(AtomicBool::new(false));
        let held_puts = Arc::new(AtomicUsize::new(0));
        let walk = Arc::new(WalkProbe::default());
        let name_hold = Arc::new(NameHold::default());
        let name_break = Arc::new(NameBreak::default());
        let credentials = Arc::new(StdMutex::new(Some(test_credentials())));
        let refused = Arc::new(StdMutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let shared = objects.clone();
        let shared_fail = fail.clone();
        let shared_requests = requests.clone();
        let shared_ranged = ranged_gets.clone();
        let shared_hold = hold_ranged_after.clone();
        let shared_hold_objects = hold_object_gets.clone();
        let shared_held_objects = held_object_gets.clone();
        let shared_hold_deletes = hold_deletes.clone();
        let shared_held_deletes = held_deletes.clone();
        let shared_hold_puts = hold_puts.clone();
        let shared_held_puts = held_puts.clone();
        let shared_walk = walk.clone();
        let shared_name_hold = name_hold.clone();
        let shared_name_break = name_break.clone();
        let shared_credentials = credentials.clone();
        let shared_refused = refused.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let objects = shared.clone();
                let fail = shared_fail.clone();
                let requests = shared_requests.clone();
                let ranged_gets = shared_ranged.clone();
                let hold_ranged_after = shared_hold.clone();
                let hold_object_gets = shared_hold_objects.clone();
                let held_object_gets = shared_held_objects.clone();
                let hold_deletes = shared_hold_deletes.clone();
                let held_deletes = shared_held_deletes.clone();
                let hold_puts = shared_hold_puts.clone();
                let held_puts = shared_held_puts.clone();
                let walk = shared_walk.clone();
                let name_hold = shared_name_hold.clone();
                let name_break = shared_name_break.clone();
                let credentials = shared_credentials.clone();
                let refused = shared_refused.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 8192];
                    let header_end;
                    loop {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(pos) = find(&buf, b"\r\n\r\n") {
                            header_end = pos + 4;
                            break;
                        }
                    }
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    let mut lines = head.lines();
                    let request_line = lines.next().unwrap_or_default().to_string();
                    let mut content_length: usize = 0;
                    let mut range_end: Option<usize> = None;
                    let mut if_none_match: Option<String> = None;
                    let mut if_match: Option<String> = None;
                    let mut content_type: Option<String> = None;
                    let mut headers: Vec<(String, String)> = Vec::new();
                    for line in lines {
                        if let Some((name, value)) = line.split_once(':') {
                            headers
                                .push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
                        }
                        let lower = line.to_ascii_lowercase();
                        if let Some(v) = lower.strip_prefix("content-length:") {
                            content_length = v.trim().parse().unwrap_or(0);
                        }
                        if let Some(v) = lower.strip_prefix("content-type:") {
                            content_type = Some(v.trim().to_string());
                        }
                        if let Some(v) = lower.strip_prefix("range:") {
                            range_end = v
                                .trim()
                                .strip_prefix("bytes=0-")
                                .and_then(|n| n.parse().ok());
                        }
                        if let Some(v) = lower.strip_prefix("if-none-match:") {
                            if_none_match = Some(v.trim().to_string());
                        }
                        if let Some(v) = lower.strip_prefix("if-match:") {
                            if_match = Some(v.trim().to_string());
                        }
                    }
                    while buf.len() < header_end + content_length {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = buf[header_end..header_end + content_length].to_vec();
                    let mut parts = request_line.split_whitespace();
                    let method = parts.next().unwrap_or_default().to_string();
                    let target = parts.next().unwrap_or_default().to_string();
                    // Every request is signed, as S3 requires of a
                    // bucket that is not public (issue #709); one
                    // that is not, or whose signature does not
                    // verify, is refused before anything else.
                    let expected = credentials.lock().unwrap().clone();
                    if let Some(creds) = expected
                        && let Err(refusal) =
                            verify_sigv4(&creds, &method, &target, &headers, &body)
                    {
                        let path = target.split_once('?').map_or(target.as_str(), |(p, _)| p);
                        let rest = path.trim_start_matches('/');
                        let key =
                            percent_decode(rest.split_once('/').map(|(_, k)| k).unwrap_or(""));
                        refused
                            .lock()
                            .unwrap()
                            .push(format!("{method} {key} {}", refusal.code));
                        let payload = if method == "HEAD" {
                            Vec::new()
                        } else {
                            refusal.document()
                        };
                        let resp = format!(
                            "HTTP/1.1 {}\r\ncontent-type: application/xml\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            refusal.status,
                            payload.len()
                        );
                        let _ = socket.write_all(resp.as_bytes()).await;
                        let _ = socket.write_all(&payload).await;
                        return;
                    }
                    if method == "DELETE" && hold_deletes.load(Ordering::SeqCst) {
                        held_deletes.fetch_add(1, Ordering::SeqCst);
                        while hold_deletes.load(Ordering::SeqCst) {
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                        held_deletes.fetch_sub(1, Ordering::SeqCst);
                    }
                    if method == "PUT" && hold_puts.load(Ordering::SeqCst) {
                        held_puts.fetch_add(1, Ordering::SeqCst);
                        while hold_puts.load(Ordering::SeqCst) {
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                        held_puts.fetch_sub(1, Ordering::SeqCst);
                    }
                    if method == "GET" && range_end.is_some() {
                        let seen = ranged_gets.fetch_add(1, Ordering::SeqCst);
                        while hold_ranged_after
                            .lock()
                            .unwrap()
                            .is_some_and(|after| seen >= after)
                        {
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                        pace_ranged_get(&walk, &target).await;
                    }
                    let listed = (method == "GET" && target.contains("list-type=2")).then(|| {
                        let query = target.split_once('?').map(|(_, query)| query).unwrap_or("");
                        let prefix = query_param(query, "prefix").unwrap_or_default();
                        if query_param(query, "continuation-token").is_none() {
                            walk.lists.lock().unwrap().push(prefix.clone());
                        }
                        let max_keys = query_param(query, "max-keys").and_then(|n| n.parse().ok());
                        walk.pages.lock().unwrap().push((prefix.clone(), max_keys));
                        prefix
                    });
                    let mut cut = false;
                    {
                        let (path, _) = match target.split_once('?') {
                            Some((p, q)) => (p, q),
                            None => (target.as_str(), ""),
                        };
                        let rest = path.trim_start_matches('/');
                        let key =
                            percent_decode(rest.split_once('/').map(|(_, k)| k).unwrap_or(""));
                        let mut line = format!("{method} {key}");
                        if if_none_match.is_some() {
                            line.push_str(" if-none-match");
                        }
                        if if_match.is_some() {
                            line.push_str(" if-match");
                        }
                        if let Some(content_type) = &content_type {
                            line.push_str(&format!(" content-type={content_type}"));
                        }
                        requests.lock().unwrap().push(line);
                        if let Some(prefix) = &listed {
                            name_hold.wait(&name_hold.lists, prefix).await;
                        } else if method == "GET" && range_end.is_none() {
                            name_hold.wait(&name_hold.reads, &key).await;
                            if NameHold::holds(&name_break.unanswered, &key) {
                                return;
                            }
                            cut = NameHold::holds(&name_break.cut, &key);
                        } else if method == "DELETE" {
                            name_hold.wait(&name_hold.deletes, &key).await;
                        }
                    }
                    let (status, payload, extra) = handle(
                        &objects,
                        &fail,
                        &method,
                        &target,
                        body,
                        range_end,
                        Conditions {
                            if_none_match: if_none_match.as_deref(),
                            if_match: if_match.as_deref(),
                        },
                    );
                    if method == "GET"
                        && range_end.is_none()
                        && !target.contains("list-type=2")
                        && hold_object_gets.load(Ordering::SeqCst)
                    {
                        held_object_gets.fetch_add(1, Ordering::SeqCst);
                        while hold_object_gets.load(Ordering::SeqCst) {
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                        held_object_gets.fetch_sub(1, Ordering::SeqCst);
                    }
                    let resp = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: {}\r\n{extra}connection: close\r\n\r\n",
                        payload.len()
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    if cut {
                        let _ = socket.write_all(&payload[..payload.len() / 2]).await;
                        return;
                    }
                    // A HEAD response advertises the object's
                    // length but carries no body.
                    if method != "HEAD" {
                        let _ = socket.write_all(&payload).await;
                    }
                });
            }
        });
        FakeS3 {
            objects,
            fail,
            requests,
            ranged_gets,
            hold_ranged_after,
            hold_object_gets,
            held_object_gets,
            hold_deletes,
            held_deletes,
            hold_puts,
            held_puts,
            walk,
            name_hold,
            name_break,
            credentials,
            refused,
            endpoint: format!("http://127.0.0.1:{port}"),
        }
    }

    /// Why the fake refused a request, in S3's error shape
    /// (issue #709).
    pub struct Refusal {
        pub status: &'static str,
        pub code: &'static str,
        message: String,
        /// More elements for the document, already escaped.
        extra: String,
    }

    impl Refusal {
        fn new(status: &'static str, code: &'static str, message: impl Into<String>) -> Self {
            Refusal {
                status,
                code,
                message: message.into(),
                extra: String::new(),
            }
        }

        /// The XML error document S3 sends with a refusal.
        pub fn document(&self) -> Vec<u8> {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error><Code>{}</Code><Message>{}</Message>{}<RequestId>FAKES3REQUEST</RequestId><HostId>fake-s3</HostId></Error>",
                self.code,
                xml_escape(&self.message),
                self.extra
            )
            .into_bytes()
        }
    }

    fn xml_escape(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    /// Check one request's SigV4 signature as S3 does (issue
    /// #709), from what arrived on the wire and nothing the
    /// client computed: the canonical request is rebuilt from
    /// the decoded path, the decoded query and the headers the
    /// request names as signed, with this module's own
    /// encoding, and the signature from `creds`' secret. The
    /// payload hash must be the body's, every `x-amz-*`
    /// header must be signed, and temporary credentials must
    /// carry their token.
    pub fn verify_sigv4(
        creds: &crate::sigv4::Credentials,
        method: &str,
        target: &str,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<(), Refusal> {
        use hmac::{KeyInit as _, Mac as _};
        use sha2::Digest as _;
        let values = |name: &str| -> Vec<&str> {
            headers
                .iter()
                .filter(|(k, _)| k == name)
                .map(|(_, v)| v.as_str())
                .collect()
        };
        let malformed = |why: &str| {
            Refusal::new(
                "400 Bad Request",
                "AuthorizationHeaderMalformed",
                format!("The authorization header is malformed; {why}"),
            )
        };
        let Some(auth) = values("authorization").first().copied() else {
            return Err(Refusal::new(
                "403 Forbidden",
                "AccessDenied",
                "Access Denied",
            ));
        };
        let Some(fields) = auth.strip_prefix("AWS4-HMAC-SHA256 ") else {
            return Err(malformed("the algorithm must be AWS4-HMAC-SHA256"));
        };
        let (mut credential, mut signed, mut signature) = (None, None, None);
        for field in fields.split(',').map(str::trim) {
            if let Some(v) = field.strip_prefix("Credential=") {
                credential = Some(v);
            } else if let Some(v) = field.strip_prefix("SignedHeaders=") {
                signed = Some(v);
            } else if let Some(v) = field.strip_prefix("Signature=") {
                signature = Some(v);
            }
        }
        let (Some(credential), Some(signed), Some(signature)) = (credential, signed, signature)
        else {
            return Err(malformed(
                "Credential, SignedHeaders and Signature are required",
            ));
        };
        let scope: Vec<&str> = credential.split('/').collect();
        if scope.len() != 5 || scope[3] != "s3" || scope[4] != "aws4_request" {
            return Err(malformed(
                "the credential scope is not <key>/<date>/<region>/s3/aws4_request",
            ));
        }
        if scope[0] != creds.access_key {
            return Err(Refusal::new(
                "403 Forbidden",
                "InvalidAccessKeyId",
                "The AWS Access Key Id you provided does not exist in our records.",
            ));
        }
        if scope[2] != FAKE_REGION {
            return Err(malformed(&format!(
                "the region '{}' is wrong; expecting '{FAKE_REGION}'",
                scope[2]
            )));
        }
        let Some(amz_date) = values("x-amz-date").first().copied() else {
            return Err(Refusal::new(
                "403 Forbidden",
                "AccessDenied",
                "AWS authentication requires a valid Date or x-amz-date header",
            ));
        };
        if amz_date.len() != 16 || amz_date.get(..8) != Some(scope[1]) {
            return Err(malformed(
                "Invalid credential date. Date is not the same as X-Amz-Date.",
            ));
        }
        let Some(payload) = values("x-amz-content-sha256").first().copied() else {
            return Err(Refusal::new(
                "400 Bad Request",
                "InvalidRequest",
                "Missing required header for this request: x-amz-content-sha256",
            ));
        };
        if payload != "UNSIGNED-PAYLOAD" && payload != hex::encode(sha2::Sha256::digest(body)) {
            return Err(Refusal::new(
                "400 Bad Request",
                "XAmzContentSHA256Mismatch",
                "The provided 'x-amz-content-sha256' header does not match what was computed.",
            ));
        }
        let token = values("x-amz-security-token").first().copied();
        if token != creds.session_token.as_deref() {
            return Err(Refusal::new(
                "400 Bad Request",
                "InvalidToken",
                "The provided token is malformed or otherwise invalid.",
            ));
        }
        let names: Vec<&str> = signed.split(';').collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted != names
            || names
                .iter()
                .any(|n| n.is_empty() || n.bytes().any(|b| b.is_ascii_uppercase()))
            || !names.contains(&"host")
        {
            return Err(malformed(
                "SignedHeaders must be sorted, lower-case, and name host",
            ));
        }
        if headers
            .iter()
            .any(|(name, _)| name.starts_with("x-amz-") && !names.contains(&name.as_str()))
        {
            return Err(Refusal::new(
                "403 Forbidden",
                "AccessDenied",
                "There were headers present in the request which were not signed",
            ));
        }
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let canonical_uri = sig_encode(&percent_decode_bytes(path, false), true);
        let mut params: Vec<(String, String)> = query
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|p| {
                let (k, v) = p.split_once('=').unwrap_or((p, ""));
                (
                    sig_encode(&percent_decode_bytes(k, true), false),
                    sig_encode(&percent_decode_bytes(v, true), false),
                )
            })
            .collect();
        params.sort();
        let canonical_query = params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let mut canonical_headers = String::new();
        for name in &names {
            let joined = values(name)
                .iter()
                .map(|v| v.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join(",");
            canonical_headers.push_str(&format!("{name}:{joined}\n"));
        }
        let canonical = format!(
            "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed}\n{payload}"
        );
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{}/{FAKE_REGION}/s3/aws4_request\n{}",
            scope[1],
            hex::encode(sha2::Sha256::digest(canonical.as_bytes()))
        );
        let mac = |key: &[u8], data: &str| {
            let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(key).unwrap();
            mac.update(data.as_bytes());
            mac.finalize().into_bytes().to_vec()
        };
        let mut key = mac(format!("AWS4{}", creds.secret_key).as_bytes(), scope[1]);
        for part in [FAKE_REGION, "s3", "aws4_request"] {
            key = mac(&key, part);
        }
        if hex::encode(mac(&key, &string_to_sign)) != signature {
            let mut refusal = Refusal::new(
                "403 Forbidden",
                "SignatureDoesNotMatch",
                "The request signature we calculated does not match the signature you provided. Check your key and signing method.",
            );
            refusal.extra = format!(
                "<AWSAccessKeyId>{}</AWSAccessKeyId><StringToSign>{}</StringToSign><SignatureProvided>{}</SignatureProvided><CanonicalRequest>{}</CanonicalRequest>",
                xml_escape(scope[0]),
                xml_escape(&string_to_sign),
                xml_escape(signature),
                xml_escape(&canonical)
            );
            return Err(refusal);
        }
        Ok(())
    }

    /// Percent-decode `text` to bytes; with `query`, a `+` is
    /// a space, as S3 reads a query string.
    fn percent_decode_bytes(text: &str, query: bool) -> Vec<u8> {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'%' if i + 2 < bytes.len() => {
                    match u8::from_str_radix(
                        std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                        16,
                    ) {
                        Ok(v) => {
                            out.push(v);
                            i += 3;
                            continue;
                        }
                        Err(_) => out.push(b'%'),
                    }
                }
                b'+' if query => out.push(b' '),
                b => out.push(b),
            }
            i += 1;
        }
        out
    }

    /// The encoding SigV4 signs: every byte but the unreserved
    /// ones as `%XX` in upper-case hex, and `/` kept in a path.
    fn sig_encode(bytes: &[u8], path: bool) -> String {
        let mut out = String::new();
        for &b in bytes {
            if b.is_ascii_alphanumeric()
                || matches!(b, b'-' | b'.' | b'_' | b'~')
                || (path && b == b'/')
            {
                out.push(b as char);
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    }

    /// The decoded value of the query parameter `name`.
    fn query_param(query: &str, name: &str) -> Option<String> {
        query.split('&').find_map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (String::from_utf8_lossy(&percent_decode_bytes(k, true)) == name)
                .then(|| String::from_utf8_lossy(&percent_decode_bytes(v, true)).into_owned())
        })
    }

    /// The continuation token of the page after `last`: opaque,
    /// with the `+`, `/` and `=` real tokens carry, so a client
    /// that does not encode them in the query fails.
    fn continuation_token(last: &str) -> String {
        format!("1+/{}=", hex::encode(last))
    }

    /// The key a [`continuation_token`] lists after.
    fn listed_after(token: &str) -> Option<String> {
        let hex_part = token.strip_prefix("1+/")?.strip_suffix('=')?;
        String::from_utf8(hex::decode(hex_part).ok()?).ok()
    }

    /// Apply the walk probe's hold and delay to one ranged
    /// GET of the object `target` names.
    async fn pace_ranged_get(walk: &WalkProbe, target: &str) {
        let path = target.split_once('?').map_or(target, |(p, _)| p);
        let rest = path.trim_start_matches('/');
        let object = percent_decode(rest.split_once('/').map(|(_, k)| k).unwrap_or(""));
        let held = || {
            walk.hold_prefix
                .lock()
                .unwrap()
                .as_deref()
                .is_some_and(|prefix| object.starts_with(prefix))
        };
        if held() {
            let now = walk.held.fetch_add(1, Ordering::SeqCst) + 1;
            walk.held_peak.fetch_max(now, Ordering::SeqCst);
            while held() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            walk.held.fetch_sub(1, Ordering::SeqCst);
        }
        let delay = walk.delay_ms.load(Ordering::SeqCst);
        if delay > 0 {
            let now = walk.delayed.fetch_add(1, Ordering::SeqCst) + 1;
            walk.delayed_peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
            walk.delayed.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    fn percent_decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).to_string()
    }

    /// The conditional request headers the fake honors.
    struct Conditions<'a> {
        if_none_match: Option<&'a str>,
        if_match: Option<&'a str>,
    }

    fn handle(
        objects: &StdMutex<HashMap<String, Vec<u8>>>,
        fail: &StdMutex<FailFlags>,
        method: &str,
        target: &str,
        body: Vec<u8>,
        range_end: Option<usize>,
        conditions: Conditions<'_>,
    ) -> (String, Vec<u8>, String) {
        let Conditions {
            if_none_match,
            if_match,
        } = conditions;
        let (path, query) = match target.split_once('?') {
            Some((p, q)) => (p, q),
            None => (target, ""),
        };
        let rest = path.trim_start_matches('/');
        let key = percent_decode(rest.split_once('/').map(|(_, k)| k).unwrap_or(""));
        let fail = fail.lock().unwrap();
        let error = (
            "500 Internal Server Error".into(),
            Vec::new(),
            String::new(),
        );
        let refusal = |code: &'static str, message: &str| {
            let refusal = Refusal::new("403 Forbidden", code, message);
            (
                refusal.status.to_string(),
                refusal.document(),
                "content-type: application/xml\r\n".to_string(),
            )
        };
        let listing = method == "GET" && query.contains("list-type=2");
        let redirect = if listing {
            &fail.redirect_listings
        } else {
            &fail.redirect_objects
        };
        if let Some(base) = redirect {
            let redirect = Refusal::new(
                "307 Temporary Redirect",
                "TemporaryRedirect",
                "Please re-send this request to the specified temporary endpoint.",
            );
            return (
                redirect.status.to_string(),
                redirect.document(),
                format!("location: {base}{target}\r\ncontent-type: application/xml\r\n"),
            );
        }
        match (method, query.contains("list-type=2")) {
            ("PUT", _) if fail.put => error,
            ("PUT", _)
                if if_none_match == Some("*") && objects.lock().unwrap().contains_key(&key) =>
            {
                // The conditional write's precondition: a
                // name already taken is a 412, nothing is
                // written.
                ("412 Precondition Failed".into(), Vec::new(), String::new())
            }
            ("PUT", _) => {
                objects.lock().unwrap().insert(key, body);
                ("200 OK".into(), Vec::new(), String::new())
            }
            ("GET", true) if fail.list => error,
            ("GET", true) => {
                let prefix = query_param(query, "prefix").unwrap_or_default();
                // Pages of at most `max-keys` names, 1000 at
                // most and by default, as S3 lists them (issue
                // #709), each naming the next by an opaque
                // continuation token.
                let max_keys = query_param(query, "max-keys")
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(1000)
                    .clamp(1, 1000);
                let after = match query_param(query, "continuation-token") {
                    None => None,
                    Some(token) => match listed_after(&token) {
                        Some(after) => Some(after),
                        None => {
                            let refusal = Refusal::new(
                                "400 Bad Request",
                                "InvalidArgument",
                                "The continuation token provided is incorrect",
                            );
                            return (
                                refusal.status.to_string(),
                                refusal.document(),
                                "content-type: application/xml\r\n".to_string(),
                            );
                        }
                    },
                };
                // Keys list in ascending order, as S3 lists
                // them, so a test can place objects ahead of
                // others in a purge walk.
                let mut keys: Vec<(String, String)> = objects
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(k, _)| k.starts_with(&prefix))
                    .filter(|(k, _)| {
                        after
                            .as_ref()
                            .is_none_or(|after| k.as_str() > after.as_str())
                    })
                    .map(|(k, v)| (k.clone(), etag_of(v)))
                    .collect();
                keys.sort();
                let truncated = keys.len() > max_keys || fail.stuck_token;
                keys.truncate(max_keys);
                let mut xml = format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><KeyCount>{}</KeyCount><MaxKeys>{max_keys}</MaxKeys><IsTruncated>{truncated}</IsTruncated>",
                    keys.len()
                );
                if fail.stuck_token {
                    let token = query_param(query, "continuation-token")
                        .unwrap_or_else(|| continuation_token(""));
                    xml += &format!("<NextContinuationToken>{token}</NextContinuationToken>");
                } else if truncated && let Some((last, _)) = keys.last() {
                    xml += &format!(
                        "<NextContinuationToken>{}</NextContinuationToken>",
                        continuation_token(last)
                    );
                }
                for (k, etag) in keys {
                    // S3 escapes the ETag's quotes in listings.
                    let etag = etag.replace('"', "&quot;");
                    xml += &format!("<Contents><Key>{k}</Key><ETag>{etag}</ETag></Contents>");
                }
                xml += "</ListBucketResult>";
                ("200 OK".into(), xml.into_bytes(), String::new())
            }
            ("GET", false) if fail.refuse_get => refusal("AccessDenied", "Access Denied"),
            ("GET", false) if fail.get_object => error,
            ("GET", false) => match objects.lock().unwrap().get(&key) {
                Some(v) => {
                    let etag = format!("etag: {}\r\n", etag_of(v));
                    match range_end {
                        Some(end) if end + 1 < v.len() => {
                            let total = v.len();
                            (
                                "206 Partial Content".into(),
                                v[..=end].to_vec(),
                                format!("content-range: bytes 0-{end}/{total}\r\n{etag}"),
                            )
                        }
                        _ => ("200 OK".into(), v.clone(), etag),
                    }
                }
                None => ("404 Not Found".into(), Vec::new(), String::new()),
            },
            ("HEAD", _) if fail.get_object => error,
            ("HEAD", _) => match objects.lock().unwrap().get(&key) {
                // The payload stands in for the length; the
                // connection loop writes none of it.
                Some(v) => (
                    "200 OK".into(),
                    v.clone(),
                    format!("etag: {}\r\n", etag_of(v)),
                ),
                None => ("404 Not Found".into(), Vec::new(), String::new()),
            },
            ("DELETE", _) if fail.refuse_delete => refusal("AccessDenied", "Access Denied"),
            ("DELETE", _) if fail.delete => error,
            ("DELETE", _) if fail.if_match_delete && if_match.is_some() => {
                ("501 Not Implemented".into(), Vec::new(), String::new())
            }
            ("DELETE", _) => {
                let mut objects = objects.lock().unwrap();
                match (if_match, objects.get(&key)) {
                    // A conditional delete of a missing
                    // object, or of one whose bytes changed
                    // since the ETag was read, is refused.
                    (Some(_), None) => ("404 Not Found".into(), Vec::new(), String::new()),
                    (Some(expected), Some(v)) if etag_of(v) != expected => {
                        ("412 Precondition Failed".into(), Vec::new(), String::new())
                    }
                    _ => {
                        objects.remove(&key);
                        ("204 No Content".into(), Vec::new(), String::new())
                    }
                }
            }
            _ => ("400 Bad Request".into(), Vec::new(), String::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{TAIL_FILLER, filler_name, start_fake, test_aws};
    use super::*;
    use crate::url::ImageFormat;

    /// The entry MAC key the test stores run under.
    fn test_entry_key() -> Vec<u8> {
        crate::url::entry_key(b"s3store-test-secret")
    }

    /// A store over `endpoint` that signs with the fake's
    /// credentials (issue #709): the fake refuses any request
    /// whose signature does not verify.
    fn store(endpoint: &str) -> S3Store {
        S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(endpoint.into()),
            "us-east-1".into(),
            Some(test_aws()),
            test_entry_key(),
        )
        .unwrap()
    }

    fn image(bytes: &[u8], groups: &[&str]) -> CachedImage {
        CachedImage {
            bytes: bytes::Bytes::copy_from_slice(bytes),
            format: ImageFormat::WebP,
            quality: Some(80),
            groups: groups.iter().map(|g| g.to_string()).collect(),
            etag: crate::cache::etag_for_bytes(bytes),
            written: None,
        }
    }

    /// `image`, stamped `offset_ms` from `cutoff`'s start by
    /// the purging process.
    fn stamped(bytes: &[u8], groups: &[&str], cutoff: &PurgeCutoff, offset_ms: i64) -> CachedImage {
        CachedImage {
            written: Some(crate::cache::WriteStamp {
                at_ms: cutoff.start_ms.saturating_add_signed(offset_ms),
                writer: cutoff.writer,
            }),
            ..image(bytes, groups)
        }
    }

    fn hero_cutoff() -> PurgeCutoff {
        PurgeCutoff {
            start_ms: 1_762_000_000_000,
            writer: 0xabc,
        }
    }

    #[tokio::test]
    async fn put_get_round_trip_across_instances() {
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        assert!(a.get("300x200/example.com/i.jpg").await.is_none());
        a.put("300x200/example.com/i.jpg", &image(b"pixels", &["hero"]))
            .await;
        // A second instance (another replica) reads it back.
        let got = b.get("300x200/example.com/i.jpg").await.unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from_static(b"pixels"));
        assert_eq!(got.groups, vec!["hero".to_string()]);
        assert_eq!(got.quality, Some(80));
    }

    #[tokio::test]
    async fn every_request_names_the_digest_of_the_body_it_sends() {
        // Issue #644: a request with no body names the
        // digest of nothing from a constant, and a write
        // hashes its payload, without formatting it a byte
        // at a time. Either way `x-amz-content-sha256` is
        // the lower-case hex SHA-256 of the body the store
        // receives, which the signature covers.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        assert_eq!(
            sigv4::EMPTY_PAYLOAD_SHA256,
            hex::encode(Sha256::digest(b""))
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    // One request per connection: its head,
                    // then the body its length declares.
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let head_end = loop {
                        if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break at + 4;
                        }
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&chunk[..n]),
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
                    let header = |name: &str| {
                        head.lines()
                            .find_map(|line| line.strip_prefix(name)?.strip_prefix(':'))
                            .map(|value| value.trim().to_string())
                    };
                    let length: usize = header("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    while buf.len() < head_end + length {
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&chunk[..n]),
                        }
                    }
                    let method = head.split(' ').next().unwrap_or_default().to_string();
                    // A write is accepted; a read finds nothing.
                    let status = if method == "put" {
                        "200 OK"
                    } else {
                        "404 Not Found"
                    };
                    recorded.lock().unwrap().push((
                        method,
                        header("x-amz-content-sha256"),
                        header("authorization"),
                        buf[head_end..head_end + length].to_vec(),
                    ));
                    let answer = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(answer.as_bytes()).await;
                });
            }
        });
        let store = S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(endpoint),
            "us-east-1".into(),
            Some(
                sigv4::Credentials {
                    access_key: "AKIDEXAMPLE".to_string(),
                    secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
                    session_token: None,
                }
                .into(),
            ),
            test_entry_key(),
        )
        .unwrap();
        assert!(store.get("300x200/example.com/i.jpg").await.is_none());
        store
            .put("300x200/example.com/i.jpg", &image(b"pixels", &["hero"]))
            .await;
        assert_eq!(store.errors(), 0);
        let seen = seen.lock().unwrap().clone();
        let methods: Vec<&str> = seen.iter().map(|(method, ..)| method.as_str()).collect();
        assert_eq!(methods, ["get", "put"]);
        for (method, digest, authorization, body) in &seen {
            assert_eq!(body.is_empty(), method == "get", "{method}");
            assert_eq!(
                digest.as_deref(),
                Some(hex::encode(Sha256::digest(body)).as_str()),
                "{method}"
            );
            let authorization = authorization.as_deref().unwrap_or_default();
            assert!(
                authorization.contains("signedheaders=host;x-amz-content-sha256;x-amz-date"),
                "{method}: {authorization}"
            );
        }
        assert_eq!(seen[0].1.as_deref(), Some(sigv4::EMPTY_PAYLOAD_SHA256));
    }

    #[tokio::test]
    async fn tier_failures_are_counted_not_silent() {
        // Issue #367: an unreachable store makes every
        // operation fail at transport. The results stay
        // best-effort (the put drops, the get misses), but
        // each failure lands on the counter the metrics
        // exposition reads.
        let store = store("http://127.0.0.1:1");
        assert_eq!(store.errors(), 0);
        store.put("k", &image(b"payload", &[])).await;
        assert_eq!(store.errors(), 1);
        assert!(store.get("k").await.is_none());
        assert_eq!(store.errors(), 2);
    }

    #[tokio::test]
    async fn raw_puts_carry_the_sniffed_content_type() {
        // Issue #366: both raw write paths record the
        // content type the bytes sniff as, so a store whose
        // default for headerless objects is a binary type
        // still serves the upload back as an image. The
        // cache entry path is unchanged: its objects are
        // framed entries, not originals.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[0u8; 16]);
        assert!(s.put_raw("photo.png", png.clone()).await);
        assert_eq!(s.put_raw_if_absent("photo2.png", png).await, RawPut::Stored);
        s.put("key", &image(b"pixels", &[])).await;
        let requests = fake.requests.lock().unwrap().clone();
        let raw_lines: Vec<&String> = requests.iter().filter(|r| r.contains("photo")).collect();
        assert_eq!(raw_lines.len(), 2, "{requests:?}");
        for line in raw_lines {
            assert!(line.contains("content-type=image/png"), "{line}");
        }
        assert!(
            requests
                .iter()
                .all(|r| !r.contains(".bin") || !r.contains("content-type")),
            "{requests:?}"
        );
    }

    #[tokio::test]
    async fn remove_deletes_exactly_one_entry() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("one", &image(b"1", &[])).await;
        s.put("one#avif", &image(b"2", &[])).await;
        s.remove("one").await;
        assert!(s.get("one").await.is_none());
        // The variant is a different key: untouched.
        assert!(s.get("one#avif").await.is_some());
        // Removing an absent key is a quiet no-op.
        s.remove("one").await;
        s.remove("missing").await;
    }

    #[tokio::test]
    async fn purge_matching_removes_the_base_and_its_variants_only() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        for key in [
            "a/example.com/i.jpg",
            "a/example.com/i.jpg#w2",
            "ab/example.com/i.jpg",
        ] {
            s.put(key, &image(b"x", &[])).await;
        }
        assert_eq!(
            s.purge_matching("a/example.com/i.jpg", "a/example.com/i.jpg")
                .await,
            2
        );
        assert!(s.get("a/example.com/i.jpg").await.is_none());
        assert!(s.get("ab/example.com/i.jpg").await.is_some());
    }

    #[tokio::test]
    async fn a_bounded_purge_scans_part_and_finishes_from_the_pending_rest() {
        // Issue #368: with a scan budget below the object
        // count, the inline call reads only the budgeted
        // headers and hands back the un-scanned remainder;
        // finishing it removes exactly the matches the
        // whole walk would have removed.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        for i in 0..5 {
            s.put(&format!("k{i}"), &image(b"x", &["hero"])).await;
        }
        s.put("other", &image(b"x", &["side"])).await;
        let hero = PurgeTarget::Group("hero".into());
        let (removed, pending) = s
            .purge_bounded(&hero, None, PurgeCutoff::EVERYTHING, 2)
            .await;
        assert!(removed.complete);
        assert!(removed.removed <= 2, "inline removals bounded: {removed:?}");
        let pending = pending.expect("a remainder past the budget");
        let rest = s.finish_purge(pending).await.removed;
        assert_eq!(removed.removed + rest, 5);
        assert!(s.get("other").await.is_some());
        // A budget above the object count leaves nothing
        // pending.
        s.put("k9", &image(b"x", &["hero"])).await;
        let (removed, pending) = s
            .purge_bounded(&hero, None, PurgeCutoff::EVERYTHING, 128)
            .await;
        assert_eq!(
            removed,
            Swept {
                removed: 1,
                complete: true
            }
        );
        assert!(pending.is_none());
    }

    #[tokio::test]
    async fn a_walk_leaves_and_does_not_count_copies_written_after_the_purge_began() {
        // Issue #446: the walk removes the copies written
        // before the purge began (and unstamped ones, which
        // predate every purge) and leaves fresh renders in
        // place, uncounted, inline and in the remainder alike.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        let cutoff = hero_cutoff();
        s.put("old", &stamped(b"1", &["hero"], &cutoff, -1)).await;
        s.put("legacy", &image(b"2", &["hero"])).await;
        s.put("fresh", &stamped(b"3", &["hero"], &cutoff, 0)).await;
        s.put("fresher", &stamped(b"4", &["hero"], &cutoff, 60_000))
            .await;
        // Another replica's copy from just after the purge
        // began is inside the skew margin: purged.
        let mut foreign = stamped(b"5", &["hero"], &cutoff, 1_000);
        foreign.written.as_mut().unwrap().writer ^= 1;
        s.put("foreign", &foreign).await;
        let hero = PurgeTarget::Group("hero".into());
        let (inline, pending) = s.purge_bounded(&hero, None, cutoff, 2).await;
        let rest = s.finish_purge(pending.expect("a remainder")).await;
        assert_eq!(inline.removed + rest.removed, 3);
        assert!(s.get("old").await.is_none());
        assert!(s.get("legacy").await.is_none());
        assert!(s.get("foreign").await.is_none());
        assert!(s.get("fresh").await.is_some());
        assert!(s.get("fresher").await.is_some());
        // A per-URL purge's direct variant reads apply the
        // same rule.
        let base = "30x20/http://x/a.jpg";
        let epoch = format!("{base}#cfgtest");
        s.put(&epoch, &stamped(b"6", &[], &cutoff, -5)).await;
        s.put(&format!("{epoch}#avif"), &stamped(b"7", &[], &cutoff, 5))
            .await;
        let key = PurgeTarget::Key(base.into());
        let (removed, _) = s.purge_bounded(&key, Some(&epoch), cutoff, 128).await;
        assert_eq!(removed.removed, 1);
        assert!(s.get(&epoch).await.is_none());
        assert!(s.get(&format!("{epoch}#avif")).await.is_some());
        // Every delete the walk sent was conditional.
        let requests = fake.requests.lock().unwrap().clone();
        let deletes: Vec<&String> = requests
            .iter()
            .filter(|r| r.starts_with("DELETE"))
            .collect();
        assert!(!deletes.is_empty());
        assert!(
            deletes.iter().all(|r| r.ends_with(" if-match")),
            "{deletes:?}"
        );
    }

    /// Fill the fake bucket with `count` filler objects under
    /// the test store's prefix, `cache/`, each named after
    /// [`TAIL_FILLER`] (see [`filler_name`]): a walk reads each
    /// one's header and leaves it.
    fn seed_fillers(fake: &super::test_support::FakeS3, count: usize) {
        let mut objects = fake.objects.lock().unwrap();
        for i in 0..count {
            objects.insert(
                format!("cache/{}", filler_name(TAIL_FILLER, i)),
                b"not a cache entry".to_vec(),
            );
        }
    }

    /// Poll `done` every 10 ms for up to five seconds, or
    /// fail with `what`.
    async fn wait_for(what: &str, done: impl Fn() -> bool) {
        for _ in 0..500 {
            if done() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("timed out waiting until {what}");
    }

    #[tokio::test]
    async fn a_walk_reads_sixteen_headers_at_a_time() {
        // Issue #462: the walk read one entry header after
        // another, each a round trip to the store. It now
        // keeps PURGE_READS_AT_ONCE reads in flight, and no
        // more: with every read parked, exactly that many
        // arrive, and the rest wait for a free place.
        use std::sync::atomic::Ordering::SeqCst;
        let fake = start_fake().await;
        let s = Arc::new(store(&fake.endpoint));
        seed_fillers(&fake, 64);
        *fake.walk.hold_prefix.lock().unwrap() = Some(format!("cache/{TAIL_FILLER}"));
        let walker = s.clone();
        let walk = tokio::spawn(async move { walker.purge_group("hero").await });
        wait_for("the walk parks its reads", || {
            fake.walk.held.load(SeqCst) >= PURGE_READS_AT_ONCE
        })
        .await;
        // Nothing else arrives while those are parked.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert_eq!(fake.walk.held.load(SeqCst), PURGE_READS_AT_ONCE);
        assert_eq!(fake.ranged_gets.load(SeqCst), PURGE_READS_AT_ONCE);
        *fake.walk.hold_prefix.lock().unwrap() = None;
        assert_eq!(walk.await.unwrap(), 0);
        assert_eq!(fake.ranged_gets.load(SeqCst), 64);
        assert_eq!(fake.walk.held_peak.load(SeqCst), PURGE_READS_AT_ONCE);
    }

    #[tokio::test]
    async fn a_walk_over_n_objects_takes_about_n_round_trips_over_sixteen() {
        // Issue #462, against a bucket that answers each
        // header read after a delay standing in for the
        // store's round trip: one read at a time, a walk
        // over N objects took N round trips. Sixteen at a
        // time, it can take no fewer than N / 16 (no more
        // than sixteen are ever in flight), and it takes
        // about that many, far fewer than N.
        use std::sync::atomic::Ordering::SeqCst;
        const N: u64 = 160;
        const RTT_MS: u64 = 40;
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        seed_fillers(&fake, N as usize);
        fake.walk.delay_ms.store(RTT_MS, SeqCst);
        let started = std::time::Instant::now();
        assert_eq!(s.purge_group("hero").await, 0);
        let took = started.elapsed();
        assert_eq!(fake.ranged_gets.load(SeqCst), N as usize);
        assert!(fake.walk.delayed_peak.load(SeqCst) <= PURGE_READS_AT_ONCE);
        let floor = std::time::Duration::from_millis(N / 16 * RTT_MS);
        let one_at_a_time = std::time::Duration::from_millis(N * RTT_MS);
        assert!(took >= floor, "{took:?} is under {floor:?}");
        // Within four times the ideal, a quarter of the
        // one-at-a-time walk: slack for a loaded machine.
        assert!(
            took < one_at_a_time / 4,
            "{took:?} against {one_at_a_time:?}"
        );
    }

    /// A running purge of `group` for the walk registry
    /// tests.
    fn group_purge(guard: &Arc<PurgeGuard>, group: &str) -> SharedPurge {
        SharedPurge {
            active: guard.begin(PurgeTarget::Group(group.into())),
            tombstone: None,
            disk: None,
            event: "purge-group",
            log_target: format!("purge-group/{group}"),
        }
    }

    #[test]
    fn identical_purges_join_the_running_walk_until_it_ends() {
        // Issue #462: a purge whose target already has a walk
        // running here joins it instead of walking again; a
        // different target walks on its own; and once the
        // leader finds nobody waiting, the target is free.
        let s = Arc::new(store("http://127.0.0.1:1"));
        let guard = Arc::new(PurgeGuard::new());
        let (mut lease, first) = s.lead_or_join(group_purge(&guard, "hero")).unwrap();
        assert!(s.lead_or_join(group_purge(&guard, "hero")).is_none());
        assert!(s.lead_or_join(group_purge(&guard, "hero")).is_none());
        let (mut side_lease, side) = s.lead_or_join(group_purge(&guard, "side")).unwrap();
        assert_eq!(guard.active_count(), 4);
        // The leader takes over both joiners at once, and the
        // target stays taken while it walks for them.
        let joined = lease.take_joined();
        assert_eq!(joined.len(), 2);
        assert!(s.lead_or_join(group_purge(&guard, "hero")).is_none());
        assert_eq!(lease.take_joined().len(), 1);
        // Nobody left waiting: the walk is over, and the next
        // identical purge leads a walk of its own.
        assert!(lease.take_joined().is_empty());
        drop(lease);
        let (mut again, next) = s.lead_or_join(group_purge(&guard, "hero")).unwrap();
        assert!(again.take_joined().is_empty());
        assert!(side_lease.take_joined().is_empty());
        drop((first, side, joined, next));
        assert_eq!(guard.active_count(), 0);
    }

    #[test]
    fn a_dropped_walk_lease_frees_its_target_and_ends_its_joiners() {
        // Issue #462: if the task walking for a target dies
        // (a panic, a runtime teardown), the target must not
        // stay taken, or every later identical purge would
        // join a walk that never comes. The joined purges end
        // with the lease; their bucket tombstones, never
        // marked ended, keep them in force.
        let s = Arc::new(store("http://127.0.0.1:1"));
        let guard = Arc::new(PurgeGuard::new());
        let (lease, first) = s.lead_or_join(group_purge(&guard, "hero")).unwrap();
        assert!(s.lead_or_join(group_purge(&guard, "hero")).is_none());
        assert_eq!(guard.active_count(), 2);
        drop(lease);
        assert_eq!(guard.active_count(), 1);
        drop(first);
        assert!(s.lead_or_join(group_purge(&guard, "hero")).is_some());
    }

    #[tokio::test]
    async fn a_walk_the_store_does_not_answer_is_incomplete() {
        // Issue #462 review: a walk whose listing, header
        // reads or deletes fail cannot have reached every
        // covered copy, and must say so, so that its purge's
        // tombstone stays in force instead of being marked
        // ended. Each failure is counted for /metrics.
        // Objects that are not entries are answers, not
        // failures.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        seed_fillers(&fake, 4);
        s.put("old", &image(b"1", &["hero"])).await;
        let hero = PurgeTarget::Group("hero".into());
        let errors = s.errors();
        let incomplete = Swept {
            removed: 0,
            complete: false,
        };
        let failing: [fn(&mut super::test_support::FailFlags, bool); 3] = [
            |fail, on| fail.list = on,
            |fail, on| fail.get_object = on,
            |fail, on| fail.delete = on,
        ];
        for (n, switch) in failing.iter().enumerate() {
            switch(&mut fake.fail.lock().unwrap(), true);
            let (swept, _) = s
                .purge_bounded(&hero, None, PurgeCutoff::EVERYTHING, usize::MAX)
                .await;
            switch(&mut fake.fail.lock().unwrap(), false);
            assert_eq!(swept, incomplete, "failure {n}");
            assert_eq!(s.errors(), errors + n as u64 + 1, "failure {n}");
            assert!(s.get("old").await.is_some(), "failure {n}");
        }
        let (swept, _) = s
            .purge_bounded(&hero, None, PurgeCutoff::EVERYTHING, usize::MAX)
            .await;
        assert_eq!(
            swept,
            Swept {
                removed: 1,
                complete: true
            }
        );
        assert_eq!(s.errors(), errors + 3);
    }

    #[tokio::test]
    async fn a_walk_spares_a_copy_replaced_between_its_read_and_its_delete() {
        // Issue #446 (and the older #368 race): a fresh
        // render that lands under the object name the walk
        // just read must survive its delete and not count.
        let fake = start_fake().await;
        let s = std::sync::Arc::new(store(&fake.endpoint));
        let cutoff = hero_cutoff();
        s.put("k", &stamped(b"old", &["hero"], &cutoff, -1)).await;
        fake.hold_deletes
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let walker = s.clone();
        let walk = tokio::spawn(async move {
            let hero = PurgeTarget::Group("hero".into());
            walker
                .purge_bounded(&hero, None, cutoff, 128)
                .await
                .0
                .removed
        });
        for _ in 0..500 {
            if fake.held_deletes.load(std::sync::atomic::Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            fake.held_deletes.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        // A post-purge render overwrites the object while
        // the delete is in flight.
        s.put("k", &stamped(b"new", &["hero"], &cutoff, 10)).await;
        fake.hold_deletes
            .store(false, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(walk.await.unwrap(), 0);
        assert_eq!(
            s.get("k").await.unwrap().bytes,
            bytes::Bytes::from_static(b"new")
        );
    }

    #[tokio::test]
    async fn remove_if_match_removes_only_the_copy_it_read() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("k", &image(b"one", &[])).await;
        let (_, etag) = s.get_tagged("k").await.unwrap();
        let etag = etag.expect("the store reports an ETag");
        s.put("k", &image(b"two", &[])).await;
        s.remove_if_match("k", Some(&etag)).await;
        assert!(s.get("k").await.is_some(), "a replaced copy was removed");
        let (_, etag) = s.get_tagged("k").await.unwrap();
        s.remove_if_match("k", etag.as_deref()).await;
        assert!(s.get("k").await.is_none());
    }

    #[tokio::test]
    async fn a_store_that_refuses_conditional_deletes_still_purges() {
        // A store without conditional deletes may answer an
        // If-Match DELETE with 501 instead of ignoring the
        // header. Purges must still remove the copies they
        // found purged, in the walk and on sight, or nothing
        // would ever leave such a store.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        fake.fail.lock().unwrap().if_match_delete = true;
        let cutoff = hero_cutoff();
        s.put("old", &stamped(b"1", &["hero"], &cutoff, -1)).await;
        s.put("fresh", &stamped(b"2", &["hero"], &cutoff, 1)).await;
        let hero = PurgeTarget::Group("hero".into());
        let (removed, pending) = s.purge_bounded(&hero, None, cutoff, 128).await;
        assert_eq!(removed.removed, 1);
        assert!(pending.is_none());
        assert!(s.get("old").await.is_none());
        assert!(s.get("fresh").await.is_some());
        s.put("k", &image(b"one", &[])).await;
        let (_, etag) = s.get_tagged("k").await.unwrap();
        s.remove_if_match("k", etag.as_deref()).await;
        assert!(s.get("k").await.is_none());
        // Each removal was tried with the precondition first.
        let requests = fake.requests.lock().unwrap().clone();
        let conditional = requests
            .iter()
            .filter(|r| r.starts_with("DELETE") && r.ends_with(" if-match"))
            .count();
        assert_eq!(conditional, 2, "{requests:?}");
    }

    fn tombstone_for(group: &str, cutoff: PurgeCutoff, ended_ms: Option<u64>) -> Tombstone {
        Tombstone {
            target: PurgeTarget::Group(group.into()),
            cutoff,
            ended_ms,
        }
    }

    #[tokio::test]
    async fn tombstones_reach_another_replica_and_the_walk_skips_them() {
        // Issue #461: a purge's tombstone, written by one
        // replica, covers another replica's copies once it
        // refreshes, and the purge walk never reads or
        // deletes it.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xa,
        };
        let name = a
            .put_tombstone(&tombstone_for("hero", cutoff, None), 1)
            .await
            .expect("tombstone stored");
        assert!(name.starts_with("cache/purges/"), "{name}");
        let guard = PurgeGuard::new();
        assert!(b.refresh_tombstones(&guard).await);
        assert!(guard.running_covers("k", &["hero".to_string()], None));
        assert!(!guard.running_covers("k", &["side".to_string()], None));
        // Neither the walk nor its count sees it: it is
        // never read, deleted, or counted.
        let gets_before = count_gets(&fake, &name);
        s_put_hero(&a).await;
        assert_eq!(a.purge_group("hero").await, 1);
        assert!(fake.objects.lock().unwrap().contains_key(&name));
        assert_eq!(count_gets(&fake, &name), gets_before);
        // A second refresh reuses what it read: no new GET.
        assert!(b.refresh_tombstones(&guard).await);
        assert_eq!(count_gets(&fake, &name), gets_before);
        // Marking the walk's end rewrites the record, which
        // the next refresh picks up by its new ETag; the
        // tombstone still covers during its grace.
        let ended = tombstone_for("hero", cutoff, Some(now_ms()));
        assert!(a.write_tombstone(&name, &ended).await);
        assert!(b.refresh_tombstones(&guard).await);
        assert_eq!(count_gets(&fake, &name), gets_before + 1);
        assert!(guard.running_covers("k", &["hero".to_string()], None));
    }

    async fn s_put_hero(s: &S3Store) {
        s.put("member", &image(b"m", &["hero"])).await;
    }

    fn count_gets(fake: &super::test_support::FakeS3, object: &str) -> usize {
        let line = format!("GET {object}");
        fake.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| **r == line)
            .count()
    }

    #[tokio::test]
    async fn spent_tombstones_are_collected_and_live_ones_kept() {
        // Issue #461: a tombstone whose grace has passed, or
        // that a finished walk of the same target
        // supersedes, is deleted by whichever replica
        // refreshes next; a walk that never ended keeps its
        // tombstone until then.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        let now = now_ms();
        let at = |start_ms| PurgeCutoff {
            start_ms,
            writer: 0xa,
        };
        let expired = s
            .put_tombstone(
                &tombstone_for(
                    "old",
                    at(now - 2 * crate::cache::TOMBSTONE_GRACE_MS),
                    Some(now - crate::cache::TOMBSTONE_GRACE_MS - 1),
                ),
                1,
            )
            .await
            .unwrap();
        let crashed = s
            .put_tombstone(&tombstone_for("hero", at(now - 10_000), None), 2)
            .await
            .unwrap();
        let repurged = s
            .put_tombstone(&tombstone_for("hero", at(now - 5_000), Some(now)), 3)
            .await
            .unwrap();
        let unrelated = s
            .put_tombstone(&tombstone_for("side", at(now - 60_000), None), 4)
            .await
            .unwrap();
        let guard = PurgeGuard::new();
        assert!(s.refresh_tombstones(&guard).await);
        let objects = fake.objects.lock().unwrap().clone();
        assert!(!objects.contains_key(&expired));
        assert!(!objects.contains_key(&crashed));
        assert!(objects.contains_key(&repurged));
        assert!(objects.contains_key(&unrelated));
        assert!(guard.running_covers("k", &["hero".to_string()], None));
        assert!(guard.running_covers("k", &["side".to_string()], None));
        assert!(!guard.running_covers("k", &["old".to_string()], None));
    }

    #[tokio::test]
    async fn hostile_tombstone_objects_are_ignored() {
        // Hostile input: the bucket is operator-adjacent,
        // not trusted. Plants without the secret, records
        // moved to another name, oversized and malformed
        // objects, and foreign names are ignored (and never
        // deleted); a listing failure keeps the previous set
        // instead of letting a purge lapse.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xa,
        };
        let real = s
            .put_tombstone(&tombstone_for("hero", cutoff, None), 1)
            .await
            .unwrap();
        let mut other_secret = store(&fake.endpoint);
        other_secret.entry_key = crate::url::entry_key(b"another deployment").into();
        let planted = other_secret
            .put_tombstone(&tombstone_for("victim", cutoff, None), 2)
            .await
            .unwrap();
        let record = fake.objects.lock().unwrap()[&real].clone();
        let moved = format!(
            "cache/purges/{}-{:016x}-9",
            cutoff.start_ms + 1,
            cutoff.writer
        );
        let junk = [
            (moved.clone(), record.clone()),
            ("cache/purges/garbage".to_string(), record.clone()),
            (
                format!("cache/purges/1-{:016x}-1", 7),
                vec![b'x'; MAX_TOMBSTONE_BYTES + 1],
            ),
            (
                format!("cache/purges/2-{:016x}-1", 7),
                b"CRABPURGE1\nnot a record".to_vec(),
            ),
        ];
        for (name, data) in &junk {
            fake.objects
                .lock()
                .unwrap()
                .insert(name.clone(), data.clone());
        }
        let guard = PurgeGuard::new();
        assert!(s.refresh_tombstones(&guard).await);
        assert!(guard.running_covers("k", &["hero".to_string()], None));
        assert!(!guard.running_covers("k", &["victim".to_string()], None));
        let objects = fake.objects.lock().unwrap().clone();
        assert!(objects.contains_key(&planted));
        for (name, _) in &junk {
            assert!(objects.contains_key(name), "{name} was deleted");
        }
        // Names that are not tombstone names are never
        // fetched.
        let requests = fake.requests.lock().unwrap().clone();
        assert!(
            !requests.iter().any(|r| r == "GET cache/purges/garbage"),
            "{requests:?}"
        );
        // A failed listing keeps what the guard had.
        fake.fail.lock().unwrap().list = true;
        assert!(!s.refresh_tombstones(&guard).await);
        assert!(guard.running_covers("k", &["hero".to_string()], None));
        fake.fail.lock().unwrap().list = false;
        // A store that cannot be reached cannot record one;
        // the failure is counted, not silent.
        let unreachable = store("http://127.0.0.1:1");
        let before = unreachable.errors();
        assert!(
            unreachable
                .put_tombstone(&tombstone_for("hero", cutoff, None), 1)
                .await
                .is_none()
        );
        assert!(!unreachable.refresh_tombstones(&guard).await);
        assert_eq!(unreachable.errors(), before + 2);
    }

    #[tokio::test]
    async fn a_known_tombstone_whose_read_fails_stays_in_force_and_is_counted() {
        // Issue #549: a refresh dropped any tombstone it could
        // not read, without a count or a log line. A replica
        // re-reading one whose walk had just ended (a new
        // ETag) while the store answered 500 lost the purge.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let hero = ["hero".to_string()];
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xa,
        };
        let name = a
            .put_tombstone(&tombstone_for("hero", cutoff, None), 1)
            .await
            .unwrap();
        let guard = PurgeGuard::new();
        assert!(b.refresh_tombstones(&guard).await);
        let ended = tombstone_for("hero", cutoff, Some(now_ms()));
        assert!(a.write_tombstone(&name, &ended).await);
        fake.fail.lock().unwrap().get_object = true;
        let gets = count_gets(&fake, &name);
        let refreshed = b.refresh_tombstones(&guard).await;
        assert_eq!(count_gets(&fake, &name), gets + 1);
        // The version read before, whose walk never ended,
        // still applies, past the ended record's grace too.
        let later = now_ms() + 2 * crate::cache::TOMBSTONE_GRACE_MS;
        assert!(
            guard.running_covers_at(later, "k", &hero, None),
            "the purge lapsed"
        );
        assert_eq!(b.errors(), 1, "the failed read was not counted");
        assert!(!refreshed);
        // Issue #577: a tombstone kept as last read is in
        // force, so it is not unread.
        assert_eq!(b.tombstones_unread(), 0, "a kept version counted as unread");
        // It stays until a read succeeds, and every refresh
        // reads it again.
        assert!(!b.refresh_tombstones(&guard).await);
        assert_eq!(count_gets(&fake, &name), gets + 2);
        assert!(guard.running_covers_at(later, "k", &hero, None));
        assert_eq!(b.errors(), 2);
        assert_eq!(b.tombstones_unread(), 0);
        // Once the store answers, the ended record replaces
        // it, and covers during its grace only.
        fake.fail.lock().unwrap().get_object = false;
        assert!(b.refresh_tombstones(&guard).await);
        assert_eq!(count_gets(&fake, &name), gets + 3);
        assert!(guard.running_covers("k", &hero, None));
        assert!(!guard.running_covers_at(later, "k", &hero, None));
        assert!(!guard.running_covers("k", &["side".to_string()], None));
        assert_eq!(b.errors(), 2);
        assert!(b.refresh_tombstones(&guard).await);
        assert_eq!(count_gets(&fake, &name), gets + 3);
    }

    #[tokio::test]
    async fn a_known_tombstone_whose_read_breaks_off_stays_in_force_and_is_counted() {
        // Issue #549: a store fails a read without an error
        // status too. A connection that closes before any
        // answer, and a body that breaks off, fail the read
        // like a 500: the version read before stays in force,
        // each refresh counts one error and reads it again.
        let hero = ["hero".to_string()];
        for how in ["no answer", "a body that breaks off"] {
            let fake = start_fake().await;
            let a = store(&fake.endpoint);
            let b = store(&fake.endpoint);
            let cutoff = PurgeCutoff {
                start_ms: now_ms(),
                writer: 0xa,
            };
            let name = a
                .put_tombstone(&tombstone_for("hero", cutoff, None), 1)
                .await
                .unwrap();
            let guard = PurgeGuard::new();
            assert!(b.refresh_tombstones(&guard).await, "{how}");
            let ended = tombstone_for("hero", cutoff, Some(now_ms()));
            assert!(a.write_tombstone(&name, &ended).await);
            let switch = match how {
                "no answer" => &fake.name_break.unanswered,
                _ => &fake.name_break.cut,
            };
            *switch.lock().unwrap() = Some("cache/purges/".into());
            let gets = count_gets(&fake, &name);
            let later = now_ms() + 2 * crate::cache::TOMBSTONE_GRACE_MS;
            for round in 1..=2 {
                let refreshed = b.refresh_tombstones(&guard).await;
                assert_eq!(count_gets(&fake, &name), gets + round, "{how}");
                assert!(
                    guard.running_covers_at(later, "k", &hero, None),
                    "{how}: the purge lapsed"
                );
                assert_eq!(
                    b.errors(),
                    round as u64,
                    "{how}: the failed read was not counted"
                );
                assert!(!refreshed, "{how}: the failed read passed for an answer");
            }
            *switch.lock().unwrap() = None;
            assert!(b.refresh_tombstones(&guard).await, "{how}");
            assert!(guard.running_covers("k", &hero, None), "{how}");
            assert!(!guard.running_covers_at(later, "k", &hero, None), "{how}");
            assert_eq!(b.errors(), 2, "{how}");
        }
    }

    #[tokio::test]
    async fn a_tombstone_never_read_is_retried_and_counted_until_a_refresh_reads_it() {
        // Issue #549: a failed read was neither counted nor
        // logged. A tombstone this replica has never read
        // still does not apply until a refresh reads it (issue
        // #577 keeps it so; see docs/deployment.md), but every
        // refresh retries it and counts the failure. Two
        // tombstones fail in each refresh, so the count shows
        // that every failed read counts on its own, not once
        // per refresh. Issue #577: each refresh also records
        // how many listed tombstones the replica has never
        // read, and when it listed them, for /metrics.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let hero = ["hero".to_string()];
        let logo = ["logo".to_string()];
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xa,
        };
        let names = [
            a.put_tombstone(&tombstone_for("hero", cutoff, None), 1)
                .await
                .unwrap(),
            a.put_tombstone(&tombstone_for("logo", cutoff, None), 2)
                .await
                .unwrap(),
        ];
        fake.fail.lock().unwrap().get_object = true;
        let guard = PurgeGuard::new();
        assert_eq!(b.tombstones_listed_ms(), 0);
        let mut listed = 0;
        for round in 1..=2 {
            let refreshed = b.refresh_tombstones(&guard).await;
            for name in &names {
                assert_eq!(count_gets(&fake, name), round);
            }
            assert_eq!(
                b.errors(),
                2 * round as u64,
                "each failed read counts once on its own"
            );
            assert!(!refreshed);
            assert!(!guard.running_covers("k", &hero, None));
            assert!(!guard.running_covers("k", &logo, None));
            assert_eq!(b.tombstones_unread(), 2, "round {round}");
            let now_listed = b.tombstones_listed_ms();
            assert!(
                now_listed > 0 && now_listed >= listed,
                "round {round}: listed at {now_listed}, before at {listed}"
            );
            listed = now_listed;
        }
        fake.fail.lock().unwrap().get_object = false;
        assert!(b.refresh_tombstones(&guard).await);
        assert!(guard.running_covers("k", &hero, None));
        assert!(guard.running_covers("k", &logo, None));
        assert_eq!(b.errors(), 4);
        assert_eq!(b.tombstones_unread(), 0);
        assert!(b.tombstones_listed_ms() >= listed);
    }

    #[tokio::test]
    async fn a_tombstone_deleted_between_the_listing_and_its_read_is_no_failure() {
        // Issue #549: a read that answers 404 found a
        // tombstone that another replica collected after the
        // listing. That is the normal collection race, not a
        // failure: it counts no error and keeps nothing in
        // force, whether this replica read the tombstone
        // before or not.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let now = now_ms();
        let at = |start_ms| PurgeCutoff {
            start_ms,
            writer: 0xa,
        };
        let known = a
            .put_tombstone(&tombstone_for("known", at(now - 1), None), 1)
            .await
            .unwrap();
        let guard = PurgeGuard::new();
        assert!(b.refresh_tombstones(&guard).await);
        // Its walk ends, so the next refresh reads it again,
        // and one b has never read appears.
        let ended = tombstone_for("known", at(now - 1), Some(now));
        assert!(a.write_tombstone(&known, &ended).await);
        let unknown = a
            .put_tombstone(&tombstone_for("unknown", at(now), None), 2)
            .await
            .unwrap();
        *fake.name_hold.reads.lock().unwrap() = Some("cache/purges/".into());
        // Both go once the first read reaches the store, so
        // every read of them answers 404, held or not.
        let collect = async {
            let waiting = Instant::now();
            while fake
                .name_hold
                .held
                .load(std::sync::atomic::Ordering::SeqCst)
                == 0
            {
                assert!(
                    waiting.elapsed() < Duration::from_secs(30),
                    "no tombstone read reached the store"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(a.delete_object(&known).await);
            assert!(a.delete_object(&unknown).await);
            *fake.name_hold.reads.lock().unwrap() = None;
        };
        let (refreshed, ()) = tokio::join!(b.refresh_tombstones(&guard), collect);
        assert_eq!(count_gets(&fake, &known), 2);
        assert_eq!(count_gets(&fake, &unknown), 1);
        assert!(refreshed, "a collected tombstone counted as a failed read");
        assert_eq!(b.errors(), 0);
        assert!(!guard.running_covers("k", &["known".to_string()], None));
        assert!(!guard.running_covers("k", &["unknown".to_string()], None));
        // Issue #577: nor does it count as unread.
        assert_eq!(b.tombstones_unread(), 0);
    }

    #[tokio::test]
    async fn hanging_tombstone_reads_end_at_the_limit_sixteen_at_a_time() {
        // Issue #549: a refresh read the tombstones one at a
        // time, each up to the store's 10 s request bound, so
        // a boot over N tombstones waited up to N x 10 s while
        // the store hung. The reads now run
        // TOMBSTONE_READS_AT_ONCE at a time, the refresh ends
        // at its limit, and a tombstone read before keeps
        // applying.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let start = now_ms() - 60_000;
        let at = |i: usize| PurgeCutoff {
            start_ms: start + i as u64,
            writer: 0xa,
        };
        let known = a
            .put_tombstone(&tombstone_for("g0", at(0), None), 0)
            .await
            .unwrap();
        let guard = PurgeGuard::new();
        assert!(b.refresh_tombstones(&guard).await);
        // Its walk ends, so the next refresh reads it again.
        let ended = tombstone_for("g0", at(0), Some(now_ms()));
        assert!(a.write_tombstone(&known, &ended).await);
        let count = TOMBSTONE_READS_AT_ONCE + 4;
        for i in 1..count {
            a.put_tombstone(&tombstone_for(&format!("g{i}"), at(i), None), i as u64)
                .await
                .unwrap();
        }
        *fake.name_hold.reads.lock().unwrap() = Some("cache/purges/".into());
        let began = Instant::now();
        // The limit leaves the listing time to finish on a
        // loaded runner. The outer bound only turns a refresh
        // that waits on the held reads past its limit into a
        // failure.
        let refreshed = tokio::time::timeout(
            Duration::from_secs(8),
            b.refresh_tombstones_within(&guard, Duration::from_secs(2)),
        )
        .await
        .expect("the refresh outlived its limit");
        let took = began.elapsed();
        assert!(took < Duration::from_secs(6), "the refresh took {took:?}");
        assert!(!refreshed);
        assert_eq!(
            fake.name_hold
                .held_peak
                .load(std::sync::atomic::Ordering::SeqCst),
            TOMBSTONE_READS_AT_ONCE
        );
        // The reads the deadline left, held or never started,
        // count one error together: their number grows with
        // the tombstones, not with requests the store failed.
        assert_eq!(b.errors(), 1);
        // The version read before still applies; the ones
        // never read do not.
        let later = now_ms() + 2 * crate::cache::TOMBSTONE_GRACE_MS;
        assert!(
            guard.running_covers_at(later, "k", &["g0".to_string()], None),
            "the purge lapsed"
        );
        assert!(!guard.running_covers("k", &["g1".to_string()], None));
        // Once the store answers, the next refresh reads them.
        *fake.name_hold.reads.lock().unwrap() = None;
        assert!(b.refresh_tombstones(&guard).await);
        for i in 0..count {
            assert!(guard.running_covers("k", &[format!("g{i}")], None));
        }
        assert!(!guard.running_covers_at(later, "k", &["g0".to_string()], None));
        assert_eq!(b.errors(), 1);
    }

    #[tokio::test]
    async fn a_hanging_tombstone_listing_ends_at_the_limit_and_keeps_the_purges() {
        // Issue #549: one refresh ends within its limit, the
        // listing included. A listing that hangs ends at the
        // limit, is counted, and leaves the set the last
        // refresh read in force.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let hero = ["hero".to_string()];
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xa,
        };
        a.put_tombstone(&tombstone_for("hero", cutoff, None), 1)
            .await
            .unwrap();
        let guard = PurgeGuard::new();
        assert!(b.refresh_tombstones(&guard).await);
        let listed = b.tombstones_listed_ms();
        assert!(listed > 0);
        assert_eq!(b.tombstones_unread(), 0);
        *fake.name_hold.lists.lock().unwrap() = Some("cache/purges/".into());
        let began = Instant::now();
        // The outer bound only turns a refresh that waits on
        // the held listing past its limit into a failure.
        let refreshed = tokio::time::timeout(
            Duration::from_secs(8),
            b.refresh_tombstones_within(&guard, Duration::from_secs(1)),
        )
        .await
        .expect("the listing outlived the refresh's limit");
        let took = began.elapsed();
        assert!(took < Duration::from_secs(5), "the refresh took {took:?}");
        assert!(!refreshed);
        assert_eq!(
            fake.name_hold
                .held
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the listing was not held"
        );
        assert_eq!(b.errors(), 1);
        assert!(guard.running_covers("k", &hero, None));
        // Issue #577: the listing time stays at the last
        // complete listing, a second or more ago.
        assert_eq!(b.tombstones_listed_ms(), listed);
        assert_eq!(b.tombstones_unread(), 0);
        // Once the store answers, the next refresh lists again.
        *fake.name_hold.lists.lock().unwrap() = None;
        assert!(b.refresh_tombstones(&guard).await);
        assert!(guard.running_covers("k", &hero, None));
        assert_eq!(b.errors(), 1);
        let listed = b.tombstones_listed_ms();
        assert!(listed > 0);
        // Issue #577: a listing that runs out keeps the unread
        // count of the last complete listing, as well as its
        // time.
        let logo = a
            .put_tombstone(&tombstone_for("logo", cutoff, None), 2)
            .await
            .unwrap();
        *fake.name_break.unanswered.lock().unwrap() = Some(logo);
        assert!(!b.refresh_tombstones(&guard).await);
        assert_eq!(b.errors(), 2);
        assert_eq!(b.tombstones_unread(), 1);
        assert!(b.tombstones_listed_ms() >= listed);
        let listed = b.tombstones_listed_ms();
        *fake.name_hold.lists.lock().unwrap() = Some("cache/purges/".into());
        let refreshed = tokio::time::timeout(
            Duration::from_secs(8),
            b.refresh_tombstones_within(&guard, Duration::from_secs(1)),
        )
        .await
        .expect("the listing outlived the refresh's limit");
        assert!(!refreshed);
        assert_eq!(b.errors(), 3);
        assert_eq!(b.tombstones_unread(), 1);
        assert_eq!(b.tombstones_listed_ms(), listed);
        *fake.name_hold.lists.lock().unwrap() = None;
        *fake.name_break.unanswered.lock().unwrap() = None;
        assert!(b.refresh_tombstones(&guard).await);
        assert_eq!(b.tombstones_unread(), 0);
        assert!(guard.running_covers("k", &["logo".to_string()], None));
    }

    #[tokio::test]
    async fn a_first_listing_that_fails_reports_no_listing_until_one_completes() {
        // Issue #577: a replica whose listings of the
        // tombstones fail from boot on has none of them. Its
        // listing time stays 0 until a listing completes, so
        // /metrics shows that it never listed them, and its
        // unread count stays 0 with it.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xa,
        };
        a.put_tombstone(&tombstone_for("hero", cutoff, None), 1)
            .await
            .unwrap();
        let guard = PurgeGuard::new();
        assert_eq!(b.tombstones_listed_ms(), 0);
        fake.fail.lock().unwrap().list = true;
        for _ in 0..2 {
            assert!(!b.refresh_tombstones(&guard).await);
            assert_eq!(b.tombstones_listed_ms(), 0);
            assert_eq!(b.tombstones_unread(), 0);
        }
        assert!(!guard.running_covers("k", &["hero".to_string()], None));
        fake.fail.lock().unwrap().list = false;
        let before = now_ms();
        assert!(b.refresh_tombstones(&guard).await);
        let listed = b.tombstones_listed_ms();
        assert!(
            (before..=now_ms()).contains(&listed),
            "listed at {listed}, the refresh began at {before}"
        );
        assert_eq!(b.tombstones_unread(), 0);
        assert!(guard.running_covers("k", &["hero".to_string()], None));
    }

    #[tokio::test]
    async fn tombstones_listed_past_the_newest_the_refresh_reads_count_as_unread() {
        // Issue #572 fourth review: a refresh reads the newest
        // tombstones up to the cap and silently dropped the
        // rest. A tombstone held while a purge's disk files are
        // not deleted does not expire, so newer ones can push it
        // past the cap, where it no longer applies on a replica
        // that has not read it, a restarted one included. Those
        // count as unread, so /metrics shows them.
        use futures_util::StreamExt as _;
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let start_ms = now_ms() - 60_000;
        let past = 6;
        futures_util::stream::iter(0..MAX_SHARED_TOMBSTONES + past)
            .for_each_concurrent(16, |i| {
                let a = &a;
                async move {
                    let cutoff = PurgeCutoff {
                        start_ms: start_ms + i as u64,
                        writer: 0xa,
                    };
                    a.put_tombstone(&tombstone_for(&format!("g{i}"), cutoff, None), i as u64)
                        .await
                        .unwrap();
                }
            })
            .await;
        let guard = PurgeGuard::new();
        assert!(!b.refresh_tombstones(&guard).await);
        assert_eq!(b.tombstones_unread(), past as u64);
        assert_eq!(b.errors(), 0, "no read failed");
        assert!(!guard.running_covers("k", &["g0".to_string()], None));
        assert!(guard.running_covers("k", &[format!("g{past}")], None));
    }

    #[test]
    fn the_unread_line_names_the_oldest_three() {
        // Issue #577: the `cannot read purge tombstones` line
        // names the unread tombstones that began first, by the
        // start in their names, and counts the rest.
        let unread = |names: &[(u64, &str)]| -> Vec<(u64, String)> {
            names
                .iter()
                .map(|(start_ms, name)| (*start_ms, name.to_string()))
                .collect()
        };
        let mut five = unread(&[(5, "o5"), (2, "o2"), (4, "o4"), (1, "o1"), (3, "o3")]);
        assert_eq!(oldest_unread(&mut five), "o1, o2, o3 and 2 more");
        let mut three = unread(&[(3, "o3"), (1, "o1"), (2, "o2")]);
        assert_eq!(oldest_unread(&mut three), "o1, o2, o3");
        let mut two = unread(&[(9, "o2"), (1, "o1")]);
        assert_eq!(oldest_unread(&mut two), "o1, o2");
        // Two that began in the same millisecond go by name.
        let mut tied = unread(&[(7, "o9"), (7, "o8")]);
        assert_eq!(oldest_unread(&mut tied), "o8, o9");
    }

    #[tokio::test]
    async fn hanging_tombstone_collection_ends_at_the_limit_sixteen_at_a_time() {
        // Issue #549: a refresh deletes the spent tombstones it
        // finds, and those deletes count against its limit
        // too: one after another, each up to the store's 10 s
        // request bound, they held boot as long as the reads
        // did. They now run TOMBSTONE_READS_AT_ONCE at a time,
        // the ones that hang end at the deadline, and a later
        // refresh collects what is left.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let long_ago = now_ms() - crate::cache::TOMBSTONE_GRACE_MS - 60_000;
        let count = 2 * TOMBSTONE_READS_AT_ONCE + 8;
        for i in 0..count {
            let cutoff = PurgeCutoff {
                start_ms: long_ago + i as u64,
                writer: 0xa,
            };
            let spent = tombstone_for(&format!("g{i}"), cutoff, Some(long_ago + 1_000));
            a.put_tombstone(&spent, i as u64).await.unwrap();
        }
        *fake.name_hold.deletes.lock().unwrap() = Some("cache/purges/".into());
        let guard = PurgeGuard::new();
        let began = Instant::now();
        // The limit leaves the reads time to finish on a
        // loaded runner. The outer bound only turns a refresh
        // that waits on the held deletes past its limit into
        // a failure.
        let refreshed = tokio::time::timeout(
            Duration::from_secs(8),
            b.refresh_tombstones_within(&guard, Duration::from_secs(2)),
        )
        .await
        .expect("the collection outlived the refresh's limit");
        let took = began.elapsed();
        assert!(took < Duration::from_secs(6), "the refresh took {took:?}");
        assert!(refreshed, "an unfinished collection fails nothing");
        assert_eq!(b.errors(), 0);
        assert_eq!(
            fake.name_hold
                .held_peak
                .load(std::sync::atomic::Ordering::SeqCst),
            TOMBSTONE_READS_AT_ONCE
        );
        assert!(!guard.running_covers("k", &["g0".to_string()], None));
        let tombstones = || {
            fake.objects
                .lock()
                .unwrap()
                .keys()
                .filter(|name| name.starts_with("cache/purges/"))
                .count()
        };
        assert_eq!(tombstones(), count);
        // Once the store answers, the held deletes land and
        // the next refresh collects the rest.
        *fake.name_hold.deletes.lock().unwrap() = None;
        for _ in 0..500 {
            if fake
                .name_hold
                .held
                .load(std::sync::atomic::Ordering::SeqCst)
                == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(b.refresh_tombstones(&guard).await);
        assert_eq!(tombstones(), 0);
    }

    /// Run one refresh of `b` with a 3 s limit while the fake
    /// holds its tombstone listing for the first 2 s (issue
    /// #549). Returns how long the refresh took and what it
    /// returned.
    async fn refresh_after_a_slow_listing(
        fake: &super::test_support::FakeS3,
        b: &S3Store,
        guard: &PurgeGuard,
    ) -> (Duration, bool) {
        use std::sync::atomic::Ordering::SeqCst;
        *fake.name_hold.lists.lock().unwrap() = Some("cache/purges/".into());
        let began = Instant::now();
        let release = async {
            tokio::time::sleep(Duration::from_secs(2)).await;
            assert_eq!(fake.name_hold.held.load(SeqCst), 1, "no listing held");
            *fake.name_hold.lists.lock().unwrap() = None;
        };
        // The outer bound only keeps a refresh with no limit
        // at all from hanging the test.
        let (refreshed, ()) = tokio::join!(
            tokio::time::timeout(
                Duration::from_secs(20),
                b.refresh_tombstones_within(guard, Duration::from_secs(3)),
            ),
            release
        );
        let refreshed = refreshed.expect("the refresh ignored its limit");
        (began.elapsed(), refreshed)
    }

    #[tokio::test]
    async fn one_deadline_covers_the_tombstone_listing_and_reads_together() {
        // Issue #549: the limit bounds the whole refresh, not
        // each part of it. A listing that takes 2 s of a 3 s
        // limit leaves the hanging reads 1 s, not 3: with a
        // limit per part, a store slow in every part would
        // hold boot for the limit several times over.
        use std::sync::atomic::Ordering::SeqCst;
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let now = now_ms();
        for i in 0..4 {
            let cutoff = PurgeCutoff {
                start_ms: now - i,
                writer: 0xa,
            };
            a.put_tombstone(&tombstone_for(&format!("g{i}"), cutoff, None), i)
                .await
                .unwrap();
        }
        *fake.name_hold.reads.lock().unwrap() = Some("cache/purges/".into());
        let guard = PurgeGuard::new();
        let (took, refreshed) = refresh_after_a_slow_listing(&fake, &b, &guard).await;
        assert!(
            took < Duration::from_secs(4),
            "the reads got a limit of their own: the refresh took {took:?}"
        );
        assert!(!refreshed);
        assert_eq!(fake.name_hold.held.load(SeqCst), 4, "no reads held");
        assert_eq!(b.errors(), 1);
        *fake.name_hold.reads.lock().unwrap() = None;
    }

    #[tokio::test]
    async fn one_deadline_covers_the_tombstone_listing_and_deletes_together() {
        // Issue #549: as above, for the deletes of spent
        // tombstones. A listing that takes 2 s of a 3 s limit
        // leaves the hanging deletes what remains of it.
        use std::sync::atomic::Ordering::SeqCst;
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let long_ago = now_ms() - crate::cache::TOMBSTONE_GRACE_MS - 60_000;
        let count = TOMBSTONE_READS_AT_ONCE + 4;
        for i in 0..count {
            let cutoff = PurgeCutoff {
                start_ms: long_ago + i as u64,
                writer: 0xa,
            };
            let spent = tombstone_for(&format!("g{i}"), cutoff, Some(long_ago + 1_000));
            a.put_tombstone(&spent, i as u64).await.unwrap();
        }
        *fake.name_hold.deletes.lock().unwrap() = Some("cache/purges/".into());
        let guard = PurgeGuard::new();
        let (took, refreshed) = refresh_after_a_slow_listing(&fake, &b, &guard).await;
        assert!(
            took < Duration::from_secs(4),
            "the deletes got a limit of their own: the refresh took {took:?}"
        );
        assert!(refreshed, "an unfinished collection fails nothing");
        assert_eq!(
            fake.name_hold.held.load(SeqCst),
            TOMBSTONE_READS_AT_ONCE,
            "no deletes held"
        );
        assert_eq!(b.errors(), 0);
        *fake.name_hold.deletes.lock().unwrap() = None;
    }

    #[tokio::test]
    async fn tombstone_reads_done_by_the_deadline_apply_and_are_not_read_again() {
        // Issue #549: a refresh cut off at its limit keeps the
        // reads that finished by then, under their ETags, so
        // the next refresh reads only what is left. A slow
        // store that keeps answering catches up over several
        // refreshes instead of starting over on each, and the
        // newest tombstones, read first, apply first: here
        // the older ones' reads hang, more of them than the
        // refresh reads at once.
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        let now = now_ms();
        let mut new = Vec::new();
        for i in 0..4 {
            let cutoff = PurgeCutoff {
                start_ms: now - i,
                writer: 0xa,
            };
            let tombstone = tombstone_for(&format!("new{i}"), cutoff, None);
            new.push(a.put_tombstone(&tombstone, i).await.unwrap());
        }
        let old = TOMBSTONE_READS_AT_ONCE as u64 + 4;
        for i in 0..old {
            let cutoff = PurgeCutoff {
                start_ms: 1_000_000_000_000 + i,
                writer: 0xa,
            };
            a.put_tombstone(&tombstone_for(&format!("old{i}"), cutoff, None), i)
                .await
                .unwrap();
        }
        *fake.name_hold.reads.lock().unwrap() = Some("cache/purges/1000".into());
        let guard = PurgeGuard::new();
        let refreshed = tokio::time::timeout(
            Duration::from_secs(8),
            b.refresh_tombstones_within(&guard, Duration::from_secs(2)),
        )
        .await
        .expect("the refresh outlived its limit");
        assert!(!refreshed);
        for i in 0..4 {
            assert!(
                guard.running_covers("k", &[format!("new{i}")], None),
                "a read done by the deadline did not apply"
            );
        }
        assert!(!guard.running_covers("k", &["old0".to_string()], None));
        assert_eq!(b.errors(), 1);
        // Once the store answers, the next refresh reads the
        // rest, and only the rest.
        *fake.name_hold.reads.lock().unwrap() = None;
        let gets: Vec<usize> = new.iter().map(|name| count_gets(&fake, name)).collect();
        assert_eq!(gets, vec![1; 4]);
        assert!(b.refresh_tombstones(&guard).await);
        let again: Vec<usize> = new.iter().map(|name| count_gets(&fake, name)).collect();
        assert_eq!(again, gets, "a read done by the deadline was not kept");
        for i in 0..old {
            assert!(guard.running_covers("k", &[format!("old{i}")], None));
        }
        assert_eq!(b.errors(), 1);
    }

    #[test]
    fn tombstone_names_parse_strictly() {
        assert_eq!(
            parse_tombstone_name("1762000000000-00000000000000ab-3"),
            Some((1_762_000_000_000, 0xab))
        );
        for bad in [
            "",
            "garbage",
            "1-00000000000000ab",
            "1-00000000000000ab-3-4",
            "x-00000000000000ab-3",
            "1-0000000000000ab-3",
            "1-00000000000000ag-3",
            "1-00000000000000ab-",
            "-00000000000000ab-3",
            "99999999999999999999999-00000000000000ab-3",
            "18446744073709551616-00000000000000ab-3",
        ] {
            assert_eq!(parse_tombstone_name(bad), None, "{bad:?}");
        }
    }

    #[tokio::test]
    async fn purge_group_removes_only_its_members() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("one", &image(b"1", &["hero"])).await;
        s.put("two", &image(b"2", &["hero", "side"])).await;
        s.put("three", &image(b"3", &["side"])).await;
        assert_eq!(s.purge_group("hero").await, 2);
        assert!(s.get("three").await.is_some());
        assert_eq!(s.purge_group("hero").await, 0);
    }

    fn store_with_prefix(endpoint: &str, prefix: &str) -> S3Store {
        store_with_prefix_and_key(endpoint, prefix, test_entry_key())
    }

    /// A store under `prefix` whose entries carry MACs under
    /// `entry_key`: one deployment of several that share a
    /// bucket, each with its own secret (issue #608).
    fn store_with_prefix_and_key(endpoint: &str, prefix: &str, entry_key: Vec<u8>) -> S3Store {
        S3Store::new(
            "results".into(),
            prefix.into(),
            Some(endpoint.into()),
            "us-east-1".into(),
            Some(test_aws()),
            entry_key,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn read_errors_and_corrupt_entries_degrade_to_misses() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("key", &image(b"pixels", &[])).await;
        assert!(s.get("key").await.is_some());
        // A service error on the read is a miss, never a
        // failed lookup: the local tiers and the origin
        // remain available to the request.
        fake.fail.lock().unwrap().get_object = true;
        assert!(s.get("key").await.is_none());
        fake.fail.lock().unwrap().get_object = false;
        assert!(s.get("key").await.is_some());
        // An object whose bytes are not a serialized entry
        // reads as a miss too.
        fake.objects.lock().unwrap().insert(
            format!("cache/{}.bin", key_stem("key")),
            b"not an entry".to_vec(),
        );
        assert!(s.get("key").await.is_none());
    }

    #[tokio::test]
    async fn oversized_and_poisoned_objects_degrade_to_misses() {
        let fake = start_fake().await;
        let mut s = store(&fake.endpoint);
        s.max_entry_bytes = 256;
        // Under the ceiling, entries round trip as before.
        s.put("small", &image(b"x", &["hero"])).await;
        assert!(s.get("small").await.is_some());
        // Over the ceiling, the read refuses the object
        // instead of buffering it whole (R2-19), and the
        // purge walk skips it the same way: only the
        // readable entry is removed.
        s.put("big", &image(&vec![7u8; 500], &["hero"])).await;
        assert!(s.get("big").await.is_none());
        assert_eq!(s.purge_group("hero").await, 1);
        // A header claiming a group count far beyond the
        // object is a miss, never a panic or an allocation
        // sized by the count (R2-13b).
        fake.objects.lock().unwrap().insert(
            format!("cache/{}.bin", key_stem("evil")),
            b"CRABDISK4\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nwebp\n80\n999999999999999999\n4\n1\nevil\nx".to_vec(),
        );
        assert!(s.get("evil").await.is_none());
        // A validly framed entry with the wrong MAC (a
        // plant by someone without the secret, or an object
        // written under another secret) is a miss too
        // (R2-13 part a).
        let forged = {
            let mut data = crate::storage::serialize_entry(
                "planted",
                &image(b"forged", &[]),
                &test_entry_key(),
            );
            let last = data.len() - 1;
            data[last] ^= 0xFF;
            data
        };
        fake.objects
            .lock()
            .unwrap()
            .insert(format!("cache/{}.bin", key_stem("planted")), forged);
        assert!(s.get("planted").await.is_none());
    }

    #[tokio::test]
    async fn write_errors_are_dropped_silently() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        fake.fail.lock().unwrap().put = true;
        // The write reports nothing and stores nothing: the
        // failed put must not surface as an error (put has
        // no failure channel) nor leave a partial entry.
        s.put("key", &image(b"pixels", &[])).await;
        fake.fail.lock().unwrap().put = false;
        assert!(s.get("key").await.is_none());
        assert!(fake.objects.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn purge_errors_remove_nothing_and_report_zero() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("one", &image(b"1", &["hero"])).await;
        s.put("two", &image(b"2", &["hero"])).await;
        // A listing failure truncates the walk: nothing is
        // read, nothing deleted, count zero.
        fake.fail.lock().unwrap().list = true;
        assert_eq!(s.purge_group("hero").await, 0);
        // A per-URL purge no longer depends on the listing
        // for the enumerable variants: the epoch base
        // deletes directly even while the walk is
        // truncated (R2-24).
        assert_eq!(s.purge_matching("one", "one").await, 1);
        assert!(s.get("one").await.is_none());
        s.put("one", &image(b"1", &["hero"])).await;
        fake.fail.lock().unwrap().list = false;
        // Entry reads failing mid-walk skip those entries
        // the same way.
        fake.fail.lock().unwrap().get_object = true;
        assert_eq!(s.purge_group("hero").await, 0);
        fake.fail.lock().unwrap().get_object = false;
        // Deletes failing leave every entry in place, and
        // only successful deletes count once they recover.
        fake.fail.lock().unwrap().delete = true;
        assert_eq!(s.purge_group("hero").await, 0);
        fake.fail.lock().unwrap().delete = false;
        assert!(s.get("one").await.is_some());
        assert_eq!(s.purge_group("hero").await, 2);
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_degrades_like_a_failing_bucket() {
        // A port nothing listens on: every request fails at
        // the transport level instead of the service level.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let s = store(&format!("http://127.0.0.1:{port}"));
        assert!(s.get("key").await.is_none());
        s.put("key", &image(b"pixels", &[])).await;
        assert_eq!(s.purge_group("hero").await, 0);
        assert_eq!(s.purge_matching("key", "key").await, 0);
    }

    #[tokio::test]
    async fn prefixes_namespace_the_shared_bucket() {
        let fake = start_fake().await;
        let a = store_with_prefix(&fake.endpoint, "a/");
        let b = store_with_prefix(&fake.endpoint, "b/");
        a.put("same-key", &image(b"from-a", &["hero"])).await;
        // The same cache key under another prefix is a
        // different entry: invisible, and independently held.
        assert!(b.get("same-key").await.is_none());
        b.put("same-key", &image(b"from-b", &[])).await;
        assert_eq!(
            a.get("same-key").await.unwrap().bytes,
            bytes::Bytes::from_static(b"from-a")
        );
        assert_eq!(
            b.get("same-key").await.unwrap().bytes,
            bytes::Bytes::from_static(b"from-b")
        );
        assert!(
            fake.objects
                .lock()
                .unwrap()
                .keys()
                .all(|k| k.starts_with("a/") || k.starts_with("b/")),
            "objects land under their store's prefix"
        );
        // A purge under one prefix never reaches the other.
        assert_eq!(a.purge_group("hero").await, 1);
        assert!(a.get("same-key").await.is_none());
        assert_eq!(
            b.get("same-key").await.unwrap().bytes,
            bytes::Bytes::from_static(b"from-b")
        );
        assert_eq!(b.purge_matching("same-key", "same-key").await, 1);
        assert!(fake.objects.lock().unwrap().is_empty());
    }

    /// Prefix pairs of two deployments in one bucket where
    /// the outer prefix contains the inner one (issue #608):
    /// the bucket root around `prod`, and `img` around
    /// `img/prod`.
    const NESTED_PREFIXES: [(&str, &str); 2] = [("", "prod"), ("img", "img/prod")];

    /// The cache key both deployments of a nested pair hold,
    /// and its source.
    const NESTED_KEY: &str = "300x200/file:///heroes/cat.jpg";
    const NESTED_SOURCE: &str = "file:///heroes/cat.jpg";

    /// Two deployments of one bucket under `outer` and
    /// `inner`, each with its own secret (issue #608). Each
    /// holds an entry for [`NESTED_KEY`] in group `hero`. The
    /// inner one's entry is larger than the header probe, so
    /// a walk can match it on its header alone, without its
    /// MAC.
    async fn nested_deployments(endpoint: &str, outer: &str, inner: &str) -> (S3Store, S3Store) {
        let outer = store_with_prefix_and_key(endpoint, outer, crate::url::entry_key(b"outer"));
        let inner = store_with_prefix_and_key(endpoint, inner, crate::url::entry_key(b"inner"));
        let payload = vec![7u8; 2 * S3Store::HEADER_PROBE_BYTES];
        inner.put(NESTED_KEY, &image(&payload, &["hero"])).await;
        outer.put(NESTED_KEY, &image(b"outer", &["hero"])).await;
        (outer, inner)
    }

    #[tokio::test]
    async fn outer_prefix_purge_group_leaves_a_nested_deployments_entries() {
        // Issue #608: the walk of a store whose prefix holds
        // another deployment's prefix listed that deployment's
        // entries too, matched the large ones on their headers,
        // and deleted them. The outer purge now removes its own
        // entry only, and counts only that one.
        for (outer, inner) in NESTED_PREFIXES {
            let fake = start_fake().await;
            let (outer_store, inner_store) = nested_deployments(&fake.endpoint, outer, inner).await;
            assert_eq!(
                outer_store.purge_group("hero").await,
                1,
                "{outer:?} around {inner:?}"
            );
            assert!(outer_store.get(NESTED_KEY).await.is_none());
            assert!(
                inner_store.get(NESTED_KEY).await.is_some(),
                "{outer:?} around {inner:?}"
            );
        }
    }

    #[tokio::test]
    async fn outer_prefix_purge_matching_leaves_a_nested_deployments_entries() {
        // Issue #608: the same for a per-URL purge. The inner
        // entry has the same stored key, but it is not where
        // the outer store keeps that key.
        for (outer, inner) in NESTED_PREFIXES {
            let fake = start_fake().await;
            let (outer_store, inner_store) = nested_deployments(&fake.endpoint, outer, inner).await;
            assert_eq!(
                outer_store.purge_matching(NESTED_KEY, NESTED_KEY).await,
                1,
                "{outer:?} around {inner:?}"
            );
            assert!(outer_store.get(NESTED_KEY).await.is_none());
            assert!(
                inner_store.get(NESTED_KEY).await.is_some(),
                "{outer:?} around {inner:?}"
            );
        }
    }

    #[tokio::test]
    async fn outer_prefix_purge_source_leaves_a_nested_deployments_entries() {
        // Issue #608: the same for the source purge a replacing
        // upload runs.
        for (outer, inner) in NESTED_PREFIXES {
            let fake = start_fake().await;
            let (outer_store, inner_store) = nested_deployments(&fake.endpoint, outer, inner).await;
            assert_eq!(
                outer_store.purge_source(NESTED_SOURCE).await,
                1,
                "{outer:?} around {inner:?}"
            );
            assert!(outer_store.get(NESTED_KEY).await.is_none());
            assert!(
                inner_store.get(NESTED_KEY).await.is_some(),
                "{outer:?} around {inner:?}"
            );
        }
    }

    #[tokio::test]
    async fn purge_walk_ignores_an_entry_stored_under_another_name() {
        // Issue #608: a valid entry copied to another name of
        // the entry form, in the store's own prefix, is not
        // where this store keeps its key, and `get` never
        // serves it. No purge deletes it or counts it.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put(NESTED_KEY, &image(b"1", &["hero"])).await;
        let real = format!("cache/{}.bin", key_stem(NESTED_KEY));
        let copy = format!("cache/{}.bin", key_stem("another key"));
        let data = fake.objects.lock().unwrap()[&real].clone();
        fake.objects.lock().unwrap().insert(copy.clone(), data);
        assert_eq!(s.purge_group("hero").await, 1);
        assert_eq!(s.purge_matching(NESTED_KEY, NESTED_KEY).await, 0);
        assert_eq!(s.purge_source(NESTED_SOURCE).await, 0);
        let objects = fake.objects.lock().unwrap();
        assert!(!objects.contains_key(&real));
        assert!(objects.contains_key(&copy));
    }

    #[tokio::test]
    async fn the_purge_walk_reads_only_its_own_entry_objects() {
        // Issue #608: the walk read the first 16 KB of every
        // object under its prefix except its own tombstones: a
        // nested deployment's entries and tombstones, uploads,
        // anything. It now reads only names of the entry form,
        // `<64 hex>.bin`, directly under its prefix.
        let fake = start_fake().await;
        let (outer, inner) = nested_deployments(&fake.endpoint, "", "prod").await;
        {
            let mut objects = fake.objects.lock().unwrap();
            for name in [
                "purges/1-00000000000000ab-1",
                "prod/purges/1-00000000000000ab-1",
                "heroes/cat.jpg",
            ] {
                objects.insert(name.to_string(), b"not a cache entry".to_vec());
            }
        }
        fake.requests.lock().unwrap().clear();
        let purged = outer.purge_group("hero").await;
        let requests = fake.requests.lock().unwrap().clone();
        // A listing logs as `GET ` with no object name.
        let reads: Vec<&str> = requests
            .iter()
            .map(String::as_str)
            .filter(|r| r.starts_with("GET ") && *r != "GET ")
            .collect();
        let own = format!("GET {}.bin", key_stem(NESTED_KEY));
        assert_eq!(reads, [own.as_str()], "{requests:?}");
        assert_eq!(purged, 1);
        assert!(inner.get(NESTED_KEY).await.is_some());
    }

    #[tokio::test]
    async fn purge_source_removes_every_derivative_of_the_source() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("300x200/file:///heroes/cat.jpg", &image(b"1", &[]))
            .await;
        s.put(
            "300x200/file:///heroes/cat.jpg#cfgab12#avif",
            &image(b"2", &[]),
        )
        .await;
        s.put("600x400/file:///heroes/cat.jpg", &image(b"3", &[]))
            .await;
        s.put("300x200/file:///heroes/cat.jpg2", &image(b"4", &[]))
            .await;
        s.put("300x200/file:///heroes/dog.jpg", &image(b"5", &[]))
            .await;
        assert_eq!(s.purge_source("file:///heroes/cat.jpg").await, 3);
        assert!(s.get("300x200/file:///heroes/cat.jpg").await.is_none());
        assert!(s.get("300x200/file:///heroes/cat.jpg2").await.is_some());
        assert!(s.get("300x200/file:///heroes/dog.jpg").await.is_some());
    }

    #[tokio::test]
    async fn purge_matches_on_the_header_without_the_payload() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        // A payload far past the 16 KB header probe: the
        // entry is larger than the probe response, so the
        // walk can only have matched on the header (R2-24).
        let payload = vec![7u8; 64 * 1024];
        s.put("300x200/http://x/big.png", &image(&payload, &["hero"]))
            .await;
        let object_len = fake.objects.lock().unwrap().values().next().unwrap().len();
        assert!(
            object_len > 16 * 1024,
            "entry outgrows the probe: {object_len}"
        );
        assert_eq!(s.purge_group("hero").await, 1);
        assert!(s.get("300x200/http://x/big.png").await.is_none());
    }

    #[tokio::test]
    async fn raw_existence_is_a_head_and_raw_puts_can_be_conditional() {
        let fake = start_fake().await;
        let s = store_with_prefix(&fake.endpoint, "orig/");
        // Existence is answered with a HEAD, never a GET:
        // the object behind the name is not downloaded to
        // answer a boolean (issue #192, audit R2-23).
        assert!(!s.exists_raw("heroes/cat.jpg").await);
        assert!(s.put_raw("heroes/cat.jpg", b"pixels".to_vec()).await);
        assert!(s.exists_raw("heroes/cat.jpg").await);
        let requests = fake.requests.lock().unwrap().clone();
        assert!(
            requests
                .iter()
                .filter(|r| r.as_str() == "HEAD orig/heroes/cat.jpg")
                .count()
                == 2,
            "{requests:?}"
        );
        assert!(
            !requests.iter().any(|r| r.starts_with("GET ")),
            "{requests:?}"
        );
        // A conditional put stores the first writer and
        // refuses the second without touching its bytes
        // (audit R2-21).
        assert_eq!(
            s.put_raw_if_absent("heroes/dog.jpg", b"one".to_vec()).await,
            RawPut::Stored
        );
        assert_eq!(
            s.put_raw_if_absent("heroes/dog.jpg", b"two".to_vec()).await,
            RawPut::AlreadyExists
        );
        assert_eq!(
            s.get_raw("heroes/dog.jpg").await.as_deref(),
            Some(b"one".as_slice())
        );
        // A store failure reads as Failed, not as a store
        // or a conflict.
        fake.fail.lock().unwrap().put = true;
        assert_eq!(
            s.put_raw_if_absent("heroes/eel.jpg", b"x".to_vec()).await,
            RawPut::Failed
        );
    }

    // Store timeouts (issue #550).

    /// How the slow store stub treats each connection.
    #[derive(Clone)]
    enum Slow {
        /// Reads request bodies at this many bytes a second,
        /// then answers 200.
        Reads(u64),
        /// Reads this many request body bytes, then neither
        /// reads nor answers again.
        StallsAfter(usize),
        /// Reads the request head, then never answers.
        Hangs,
        /// Answers with this body, its length declared, sent
        /// at this many bytes a second.
        Serves(Arc<Vec<u8>>, u64),
        /// Answers with this body's length declared, sends
        /// this many bytes of it, then nothing more.
        ServesThenStalls(Arc<Vec<u8>>, usize),
    }

    /// A store stub that handles one request per connection
    /// the way `mode` says. Returns its endpoint and the
    /// request body bytes it has read, all connections
    /// together.
    async fn start_slow_store(mode: Slow) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let received = Arc::new(AtomicUsize::new(0));
        let counter = received.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mode = mode.clone();
                let received = counter.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = vec![0u8; 16 * 1024];
                    let head_end = loop {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break at + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
                    let length: usize = head
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                    let mut got = buf.len() - head_end;
                    received.fetch_add(got, Ordering::SeqCst);
                    let started = tokio::time::Instant::now();
                    let due = |bytes: usize, rate: u64| {
                        started + Duration::from_secs_f64(bytes as f64 / rate as f64)
                    };
                    match mode {
                        Slow::Hangs => std::future::pending::<()>().await,
                        Slow::StallsAfter(stall) => {
                            while got < stall.min(length) {
                                let n = socket.read(&mut chunk).await.unwrap_or(0);
                                if n == 0 {
                                    return;
                                }
                                got += n;
                                received.fetch_add(n, Ordering::SeqCst);
                            }
                            std::future::pending::<()>().await
                        }
                        Slow::Reads(rate) => {
                            while got < length {
                                let want = (length - got).min(chunk.len());
                                let n = socket.read(&mut chunk[..want]).await.unwrap_or(0);
                                if n == 0 {
                                    return;
                                }
                                got += n;
                                received.fetch_add(n, Ordering::SeqCst);
                                tokio::time::sleep_until(due(got, rate)).await;
                            }
                            let _ = socket
                                .write_all(
                                    b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                                )
                                .await;
                        }
                        Slow::Serves(body, rate) => {
                            let head = format!(
                                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                                body.len()
                            );
                            if socket.write_all(head.as_bytes()).await.is_err() {
                                return;
                            }
                            let step = (rate as usize / 8).clamp(1, chunk.len());
                            let mut sent = 0;
                            while sent < body.len() {
                                let end = (sent + step).min(body.len());
                                if socket.write_all(&body[sent..end]).await.is_err() {
                                    return;
                                }
                                sent = end;
                                tokio::time::sleep_until(due(sent, rate)).await;
                            }
                        }
                        Slow::ServesThenStalls(body, stall) => {
                            let head = format!(
                                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                                body.len()
                            );
                            if socket.write_all(head.as_bytes()).await.is_err() {
                                return;
                            }
                            let stall = stall.min(body.len());
                            if socket.write_all(&body[..stall]).await.is_err() {
                                return;
                            }
                            std::future::pending::<()>().await
                        }
                    }
                });
            }
        });
        (endpoint, received)
    }

    /// The body the slow transfer tests move: 16 MiB, which
    /// takes 12.8 s at [`SLOW_RATE`], past the old 10 s total
    /// and well inside its 26 s budget.
    const SLOW_BODY: usize = 16 * 1024 * 1024;

    /// A store that keeps moving, above the minimum rate:
    /// 1.25 MiB a second.
    const SLOW_RATE: u64 = 5 * 1024 * 1024 / 4;

    /// How late past its bound a request may end in these
    /// tests, for a loaded test host.
    const SLACK: Duration = Duration::from_secs(5);

    #[test]
    fn the_transfer_budget_scales_with_the_body_and_stays_bounded() {
        assert_eq!(transfer_budget(0), STORE_TIMEOUT);
        assert_eq!(transfer_budget(1024 * 1024), Duration::from_secs(11));
        assert_eq!(
            transfer_budget(SLOW_BODY as u64),
            Duration::from_secs(10 + 16)
        );
        // The repro in issue #550: 18 MB, which a store reading
        // 1 MB a second took 18 s over, against 27.2 s.
        assert_eq!(
            transfer_budget(18_000_000),
            Duration::from_millis(10_000 + 17_166)
        );
        // The largest entry and the largest upload.
        assert_eq!(
            transfer_budget(MAX_ENTRY_BYTES as u64),
            Duration::from_secs(10 + 512)
        );
        assert_eq!(
            transfer_budget(1024 * 1024 * 1024),
            Duration::from_secs(10 + 1024)
        );
    }

    #[tokio::test]
    async fn writes_that_keep_moving_outlast_the_short_bound() {
        // Issue #550: every store request had one 10 s total,
        // so an upload or a shared-tier write the store took
        // in steadily, but over more than 10 s, failed. Each
        // write now gets 10 s plus a second per MiB.
        let (endpoint, received) = start_slow_store(Slow::Reads(SLOW_RATE)).await;
        let s = store(&endpoint);
        let entry = image(&vec![2u8; SLOW_BODY], &[]);
        let started = Instant::now();
        let (stored, ()) = tokio::join!(
            s.put_raw("big.bin", vec![1u8; SLOW_BODY]),
            s.put("big", &entry)
        );
        let took = started.elapsed();
        assert!(stored, "the upload failed after {took:?}");
        assert_eq!(s.errors(), 0, "the shared-tier write failed after {took:?}");
        let read = received.load(std::sync::atomic::Ordering::SeqCst);
        assert!(read > 2 * SLOW_BODY, "the store read {read} bytes");
        assert!(took > STORE_TIMEOUT, "the stub paced too fast: {took:?}");
    }

    #[tokio::test]
    async fn a_write_the_store_stops_reading_fails_at_its_budget() {
        // A store that takes part of a body, then neither
        // reads nor answers: the write fails at its body's
        // budget (12 s for 2 MiB), not before and not much
        // after; the upload reports it and the shared-tier
        // write is counted.
        let body = 2 * 1024 * 1024;
        let (endpoint, received) = start_slow_store(Slow::StallsAfter(256 * 1024)).await;
        let s = store(&endpoint);
        let entry = image(&vec![2u8; body], &[]);
        let started = Instant::now();
        let (stored, ()) = tokio::join!(
            s.put_raw("stalled.bin", vec![1u8; body]),
            s.put("stalled", &entry)
        );
        let took = started.elapsed();
        assert!(!stored);
        assert_eq!(s.errors(), 1);
        assert!(received.load(std::sync::atomic::Ordering::SeqCst) >= 256 * 1024);
        let budget = transfer_budget(body as u64);
        assert!(took >= budget, "{took:?} against {budget:?}");
        assert!(took < budget + SLACK, "{took:?} against {budget:?}");
    }

    #[tokio::test]
    async fn an_entry_read_that_keeps_moving_outlasts_the_short_bound() {
        // A large entry served steadily over more than 10 s
        // is read whole: its budget follows the length the
        // store declares.
        let bytes = serialize_entry("big", &image(&vec![7u8; SLOW_BODY], &[]), &test_entry_key());
        let (endpoint, _) = start_slow_store(Slow::Serves(Arc::new(bytes), SLOW_RATE)).await;
        let s = store(&endpoint);
        let started = Instant::now();
        let got = s.get("big").await;
        let took = started.elapsed();
        assert_eq!(
            got.map(|image| image.bytes.len()),
            Some(SLOW_BODY),
            "after {took:?}"
        );
        assert!(took > STORE_TIMEOUT, "the stub paced too fast: {took:?}");
    }

    #[tokio::test]
    async fn reads_still_fail_fast_on_a_hung_store() {
        // A store that never answers fails a lookup, a HEAD
        // and a raw read at the short bound, as before.
        let (endpoint, _) = start_slow_store(Slow::Hangs).await;
        let s = store(&endpoint);
        let started = Instant::now();
        let (entry, exists, raw) = tokio::join!(s.get("key"), s.exists_raw("x"), s.get_raw("x"));
        let took = started.elapsed();
        assert!(entry.is_none());
        assert!(!exists);
        assert_eq!(raw, None);
        // The lookup's transport error is counted, and since
        // issue #709 so is the existence check's.
        assert_eq!(s.errors(), 2);
        assert!(took >= STORE_TIMEOUT, "{took:?}");
        assert!(took < STORE_TIMEOUT + SLACK, "{took:?}");
    }

    #[tokio::test]
    async fn a_small_read_that_trickles_keeps_the_short_bound() {
        // 64 bytes at 2 bytes a second: a byte every half
        // second, so the read timeout never fires, but the
        // read's budget (10 s for 64 bytes) ends it long
        // before the 32 s the body would take. The lookup
        // counts it as a tier failure, like a store that does
        // not answer: it is not a plain miss.
        let (endpoint, _) = start_slow_store(Slow::Serves(Arc::new(vec![b'x'; 64]), 2)).await;
        let s = store(&endpoint);
        let started = Instant::now();
        let (raw, entry) = tokio::join!(s.get_raw("x"), s.get("x"));
        let took = started.elapsed();
        assert_eq!(raw, None);
        assert!(entry.is_none());
        assert_eq!(s.errors(), 1, "the lookup that ran out is counted");
        assert!(took >= STORE_TIMEOUT, "{took:?}");
        assert!(took < STORE_TIMEOUT + SLACK, "{took:?}");
    }

    #[tokio::test]
    async fn an_entry_read_that_stalls_fails_at_the_idle_bound() {
        // A 16 MiB entry whose body stops after 256 KiB: its
        // budget is 26 s, but the body goes quiet, so the read
        // timeout ends it 10 s after the last byte, and the
        // lookup counts it as a tier failure.
        let bytes = serialize_entry("big", &image(&vec![7u8; SLOW_BODY], &[]), &test_entry_key());
        let (endpoint, _) =
            start_slow_store(Slow::ServesThenStalls(Arc::new(bytes), 256 * 1024)).await;
        let s = store(&endpoint);
        let started = Instant::now();
        let got = s.get("big").await;
        let took = started.elapsed();
        assert!(got.is_none());
        assert_eq!(s.errors(), 1, "the stalled lookup is counted");
        assert!(took >= STORE_TIMEOUT, "{took:?}");
        assert!(took < STORE_TIMEOUT + SLACK, "{took:?}");
    }

    /// The S3 examples' credentials, for the verifier test.
    fn s3_example_creds() -> sigv4::Credentials {
        sigv4::Credentials {
            access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        }
    }

    fn header_list(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// A store over the fake signing with `creds`.
    fn store_signing(endpoint: &str, creds: Option<sigv4::Credentials>) -> S3Store {
        S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(endpoint.into()),
            "us-east-1".into(),
            creds.map(Into::into),
            test_entry_key(),
        )
        .unwrap()
    }

    #[test]
    fn the_fakes_verifier_reads_sigv4_as_s3_does() {
        // Issue #709: the fake served every request whatever
        // its Authorization header said, and the signer was
        // proven only against the IAM example. The fake's
        // verifier rebuilds the canonical request from what
        // arrived; it accepts the AWS S3 GET object example,
        // signature and all...
        use super::test_support::verify_sigv4;
        let example = header_list(&[
            ("host", "examplebucket.s3.amazonaws.com"),
            ("range", "bytes=0-9"),
            ("x-amz-content-sha256", sigv4::EMPTY_PAYLOAD_SHA256),
            ("x-amz-date", "20130524T000000Z"),
            (
                "authorization",
                "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,\
                 SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,\
                 Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
            ),
        ]);
        let creds = s3_example_creds();
        assert!(verify_sigv4(&creds, "GET", "/test.txt", &example, b"").is_ok());
        // ...and refuses it once anything it signs changes: a
        // signed header, the path, the query, the body, or the
        // secret.
        let mut range = example.clone();
        range[1].1 = "bytes=0-10".to_string();
        for (target, headers, body, secret) in [
            ("/test.txt", &range, &b""[..], creds.secret_key.as_str()),
            ("/test2.txt", &example, b"", creds.secret_key.as_str()),
            ("/test.txt?x=1", &example, b"", creds.secret_key.as_str()),
            ("/test.txt", &example, b"", "another-secret"),
        ] {
            let mut creds = s3_example_creds();
            creds.secret_key = secret.to_string();
            let refused = verify_sigv4(&creds, "GET", target, headers, body).err();
            let refused = refused.unwrap_or_else(|| panic!("{target} {secret} verified"));
            assert_eq!(refused.code, "SignatureDoesNotMatch");
            assert_eq!(refused.status, "403 Forbidden");
            let document = String::from_utf8(refused.document()).unwrap();
            assert!(
                document.contains("<StringToSign>AWS4-HMAC-SHA256"),
                "{document}"
            );
            assert!(document.contains("<CanonicalRequest>GET"), "{document}");
        }
        let refused = verify_sigv4(&creds, "GET", "/test.txt", &example, b"payload").err();
        assert_eq!(refused.unwrap().code, "XAmzContentSHA256Mismatch");
        // An x-amz-* header the signature does not cover is
        // refused, as S3 refuses it.
        let mut unsigned = example.clone();
        unsigned.push(("x-amz-meta-note".to_string(), "x".to_string()));
        let refused = verify_sigv4(&creds, "GET", "/test.txt", &unsigned, b"").err();
        assert_eq!(refused.unwrap().code, "AccessDenied");
        let mut other = s3_example_creds();
        other.access_key = "AKIDSOMEONEELSE".to_string();
        let refused = verify_sigv4(&other, "GET", "/test.txt", &example, b"").err();
        assert_eq!(refused.unwrap().code, "InvalidAccessKeyId");
        let refused = verify_sigv4(&creds, "GET", "/test.txt", &example[..4], b"").err();
        assert_eq!(refused.unwrap().code, "AccessDenied");
    }

    #[tokio::test]
    async fn the_fake_refuses_what_does_not_verify_in_s3s_error_shape() {
        // Issue #709: every test store now signs with the test
        // credentials and the fake checks each request. A
        // request it refuses gets S3's XML error document and is
        // logged on the fake; the store counts the refusal.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("k", &image(b"x", &[])).await;
        assert!(s.get("k").await.is_some());
        assert_eq!((s.errors(), fake.refused().len()), (0, 0));
        // Unsigned: S3's AccessDenied, as XML.
        let resp = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(format!("{}/results/cache/x.bin", fake.endpoint))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::FORBIDDEN);
        assert_eq!(resp.headers()["content-type"], "application/xml");
        let body = resp.text().await.unwrap();
        assert!(
            body.starts_with("<?xml") && body.contains("<Code>AccessDenied</Code>"),
            "{body}"
        );
        // A wrong secret, or a key the store does not know.
        let mut not_the_secret = super::test_support::test_credentials();
        not_the_secret.secret_key = "not-the-secret".to_string();
        let wrong_secret = store_signing(&fake.endpoint, Some(not_the_secret));
        assert!(wrong_secret.get("k").await.is_none());
        let mut unknown_key = super::test_support::test_credentials();
        unknown_key.access_key = "AKIDUNKNOWN".to_string();
        let wrong_key = store_signing(&fake.endpoint, Some(unknown_key));
        assert!(wrong_key.get("k").await.is_none());
        let refused = fake.refused();
        assert_eq!(refused.len(), 3, "{refused:?}");
        assert!(refused[0].ends_with(" AccessDenied"), "{refused:?}");
        assert!(
            refused[1].ends_with(" SignatureDoesNotMatch"),
            "{refused:?}"
        );
        assert!(refused[2].ends_with(" InvalidAccessKeyId"), "{refused:?}");
        assert_eq!(wrong_secret.refused(Refused::Lookup), 1);
        assert_eq!(wrong_key.refused(Refused::Lookup), 1);
    }

    #[tokio::test]
    async fn every_kind_of_store_request_verifies_against_the_fake() {
        // Issue #709: every request shape the tier sends, signed
        // with temporary credentials (so the session token is
        // signed too), verifies: writes with a payload hash,
        // conditional writes and deletes, ranged and whole
        // reads, HEADs, listing pages with continuation tokens
        // (which carry `+`, `/` and `=`), and tombstones.
        let fake = start_fake().await;
        let mut temporary = super::test_support::test_credentials();
        temporary.session_token = Some("session/token+1=".to_string());
        *fake.credentials.lock().unwrap() = Some(temporary.clone());
        let s = store_signing(&fake.endpoint, Some(temporary));
        seed_fillers(&fake, 1500);
        s.put("k", &image(b"x", &["hero"])).await;
        let (_, etag) = s.get_tagged("k").await.unwrap();
        assert!(
            s.put_raw("raw/a b+c.png", b"\x89PNG\r\n\x1a\n".to_vec())
                .await
        );
        assert_eq!(
            s.put_raw_if_absent("raw/a b+c.png", b"x".to_vec()).await,
            RawPut::AlreadyExists
        );
        assert!(s.exists_raw("raw/a b+c.png").await);
        assert_eq!(s.get_raw("raw/a b+c.png").await.unwrap().len(), 8);
        let guard = Arc::new(PurgeGuard::new());
        let tombstone = guard.begin(PurgeTarget::Group("side".into()));
        let object = s
            .put_tombstone(tombstone.tombstone(), tombstone.sequence())
            .await
            .unwrap();
        assert!(s.refresh_tombstones(&guard).await);
        assert_eq!(s.purge_group("hero").await, 1);
        s.put("k", &image(b"y", &[])).await;
        s.remove_if_match("k", etag.as_deref()).await;
        s.remove("k").await;
        s.end_tombstone(&object, tombstone.tombstone()).await;
        assert!(fake.walk.pages_of("cache/") >= 2);
        assert_eq!(fake.refused(), Vec::<String>::new());
        assert_eq!(s.errors(), 0);
        // The same store with no token is refused: temporary
        // credentials must carry theirs.
        let tokenless = store(&fake.endpoint);
        assert!(tokenless.get("k").await.is_none());
        assert!(fake.refused()[0].ends_with(" InvalidToken"));
    }

    #[tokio::test]
    async fn an_unsigned_store_still_works_where_the_bucket_allows_it() {
        // With no credentials the requests go unsigned, for a
        // local development store that allows anonymous access.
        let fake = start_fake().await;
        *fake.credentials.lock().unwrap() = None;
        let s = store_signing(&fake.endpoint, None);
        s.put("k", &image(b"x", &["hero"])).await;
        assert!(s.get("k").await.is_some());
        assert_eq!(s.purge_group("hero").await, 1);
        assert_eq!(s.errors(), 0);
    }

    #[tokio::test]
    async fn a_purge_lists_one_page_inline_and_keeps_a_token_not_the_listing() {
        // Issue #709: a purge listed the whole prefix inline,
        // every page of it, and the walk that waited for its
        // turn kept every name the inline scan left: about 3.5
        // MB a queued purge at 31,000 objects. The inline scan
        // now lists one page, of at most its budget, and what
        // waits is where the listing stands: a continuation
        // token.
        use std::sync::atomic::Ordering::SeqCst;
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        seed_fillers(&fake, 2500);
        let hero = PurgeTarget::Group("hero".into());
        let (inline, pending) = s
            .purge_bounded(&hero, None, PurgeCutoff::EVERYTHING, 128)
            .await;
        assert!(inline.complete);
        assert_eq!(
            *fake.walk.pages.lock().unwrap(),
            vec![("cache/".to_string(), Some(128))]
        );
        assert_eq!(fake.ranged_gets.load(SeqCst), 128);
        let pending = pending.expect("a remainder past the budget");
        assert_eq!(pending.held_names(), 0);
        assert!(pending.next.as_deref().is_some_and(|t| t.len() < 512));
        // The rest lists on from the token, a page at a time,
        // and judges every other object once.
        let rest = s.finish_purge(pending).await;
        assert!(rest.complete);
        assert_eq!(fake.ranged_gets.load(SeqCst), 2500);
        let pages = fake.walk.pages.lock().unwrap().clone();
        assert_eq!(pages.len(), 4, "128, then 1000, 1000 and 372: {pages:?}");
        assert!(
            pages[1..]
                .iter()
                .all(|(prefix, max)| prefix == "cache/" && max.is_none())
        );
        assert_eq!(fake.walk.lists_of("cache/"), 1, "one pass");
    }

    #[tokio::test]
    async fn a_walk_page_by_page_reaches_every_covered_copy() {
        // Issue #709: with the listing taken a page at a time,
        // copies on every page are still judged and deleted, and
        // a pass over several pages is complete.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        seed_fillers(&fake, 1200);
        {
            let mut objects = fake.objects.lock().unwrap();
            for i in 0..1100 {
                objects.insert(
                    format!("cache/{}", filler_name(super::test_support::HEAD_FILLER, i)),
                    b"not a cache entry".to_vec(),
                );
            }
        }
        for i in 0..5 {
            s.put(&format!("member-{i}"), &image(b"x", &["hero"])).await;
        }
        let hero = PurgeTarget::Group("hero".into());
        let swept = s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING).await;
        assert_eq!(
            swept,
            Swept {
                removed: 5,
                complete: true
            }
        );
        assert_eq!(fake.walk.pages_of("cache/"), 3);
        for i in 0..5 {
            assert!(s.get(&format!("member-{i}")).await.is_none());
        }
    }

    #[tokio::test]
    async fn a_listing_that_never_moves_on_ends_the_walk_incomplete() {
        // Issue #709: a store that answered every page as
        // truncated, naming the page it was asked for as the
        // next, kept a walk listing that page forever, holding
        // its walk turn and growing its listing. The repeated
        // token now ends the walk, incomplete, so its purge's
        // tombstone stays in force.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        seed_fillers(&fake, 3);
        fake.fail.lock().unwrap().stuck_token = true;
        let hero = PurgeTarget::Group("hero".into());
        let swept = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING),
        )
        .await
        .expect("the walk ends");
        assert!(!swept.complete);
        assert_eq!(s.errors(), 1);
    }

    #[tokio::test]
    async fn a_refused_delete_of_a_covered_copy_leaves_the_walk_incomplete() {
        // Issue #709: a 403 on a covered copy's DELETE counted
        // as kept, with no line, so a purge whose credentials
        // lack s3:DeleteObject ended its tombstone with the copy
        // still in the bucket. A refusal now leaves the walk
        // incomplete, as a store failure does, and counts.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        seed_fillers(&fake, 3);
        s.put("old", &image(b"1", &["hero"])).await;
        fake.fail.lock().unwrap().refuse_delete = true;
        let hero = PurgeTarget::Group("hero".into());
        let swept = s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING).await;
        assert_eq!(
            swept,
            Swept {
                removed: 0,
                complete: false
            }
        );
        assert_eq!(s.refused(Refused::Purge), 1);
        assert_eq!(s.errors(), 1, "one line for the pass");
        assert!(s.get("old").await.is_some());
        // Once the store allows it, the next purge removes the
        // copy and is complete.
        fake.fail.lock().unwrap().refuse_delete = false;
        let swept = s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING).await;
        assert_eq!(
            swept,
            Swept {
                removed: 1,
                complete: true
            }
        );
    }

    #[tokio::test]
    async fn a_refused_header_read_leaves_the_walk_incomplete() {
        // Issue #709: a refused read of an entry header (a 403)
        // read as an object with nothing to purge, so the walk
        // ended complete without having judged it. It is now
        // a refusal: counted, and the walk is incomplete. A
        // missing object is still an answer.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("old", &image(b"1", &["hero"])).await;
        fake.fail.lock().unwrap().refuse_get = true;
        let hero = PurgeTarget::Group("hero".into());
        let swept = s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING).await;
        assert_eq!(
            swept,
            Swept {
                removed: 0,
                complete: false
            }
        );
        assert_eq!(s.refused(Refused::Purge), 1);
        fake.fail.lock().unwrap().refuse_get = false;
        assert!(s.get("old").await.is_some());
        // A per-URL purge's reads by name of variants that are
        // not there are answers, not refusals.
        let key = PurgeTarget::Key("30x20/http://x/a.jpg".into());
        let (swept, _) = s
            .purge_bounded(
                &key,
                Some("30x20/http://x/a.jpg#cfgtest"),
                PurgeCutoff::EVERYTHING,
                128,
            )
            .await;
        assert!(swept.complete);
        assert_eq!(s.refused(Refused::Purge), 1);
    }

    #[tokio::test]
    async fn refused_lookups_and_existence_checks_are_counted_not_misses() {
        // Issue #709: any lookup answer but a success was a
        // silent miss, and so was any HEAD that failed: a store
        // refusing every request (the 403 of expired keys) looked
        // like an empty cache. A refused lookup now counts as
        // one, and in the tier's errors; a lookup the store
        // fails (a 5xx) counts in the errors alone; a failed
        // existence check counts among those, refused or not;
        // and a 404 is still an ordinary miss.
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("k", &image(b"x", &[])).await;
        assert!(s.put_raw("photo.png", b"x".to_vec()).await);
        assert!(s.get("missing").await.is_none());
        assert!(!s.exists_raw("missing.png").await);
        assert_eq!((s.errors(), s.exists_checks_failed()), (0, 0));
        // The store stops honoring the keys.
        let mut rotated = super::test_support::test_credentials();
        rotated.access_key = "AKIDROTATED".to_string();
        *fake.credentials.lock().unwrap() = Some(rotated);
        assert!(s.get("k").await.is_none());
        assert!(!s.exists_raw("photo.png").await);
        assert_eq!(s.refused(Refused::Lookup), 1);
        assert_eq!(s.refused(Refused::Purge), 0);
        assert_eq!(s.exists_checks_failed(), 1);
        assert_eq!(s.errors(), 2);
        // It honors them again, but fails every read: errors,
        // not refusals.
        *fake.credentials.lock().unwrap() = Some(super::test_support::test_credentials());
        fake.fail.lock().unwrap().get_object = true;
        assert!(s.get("k").await.is_none());
        assert!(!s.exists_raw("photo.png").await);
        assert_eq!(s.refused(Refused::Lookup), 1);
        assert_eq!(s.exists_checks_failed(), 2);
        assert_eq!(s.errors(), 4);
    }

    #[tokio::test]
    async fn a_refusal_names_the_stores_error_code_and_nothing_else() {
        // Issue #709: the refused lookup's line says why with
        // the code S3 gives, read from the front of the body
        // and kept only when it is a short name.
        let refusal = |body: &str| {
            reqwest::Response::from(
                axum::http::Response::builder()
                    .status(403)
                    .body(body.to_string())
                    .unwrap(),
            )
        };
        let code = refusal_code(
            refusal(
                "<?xml version=\"1.0\"?><Error><Code>ExpiredToken</Code>\
                 <Message>The provided token has expired.</Message></Error>",
            ),
            Instant::now(),
        )
        .await;
        assert_eq!(code.as_deref(), Some("ExpiredToken"));
        // S3 puts the code first, and a document can run long
        // after it: a SignatureDoesNotMatch spells out the
        // canonical request, session token and all, byte by
        // byte. Only its front is read.
        let long = format!(
            "<Error><Code>SignatureDoesNotMatch</Code>\
             <CanonicalRequestBytes>{}</CanonicalRequestBytes></Error>",
            "47 45 54 ".repeat(3000)
        );
        let code = refusal_code(refusal(&long), Instant::now()).await;
        assert_eq!(code.as_deref(), Some("SignatureDoesNotMatch"));
        for body in [
            "",
            "<Error><Code>Bad\ncode</Code></Error>",
            "<Error><Code>with spaces</Code></Error>",
            &format!("<Error><Code>{}</Code></Error>", "A".repeat(65)),
            &format!("{}<Error><Code>Late</Code></Error>", " ".repeat(8192)),
        ] {
            assert_eq!(refusal_code(refusal(body), Instant::now()).await, None);
        }
    }

    #[tokio::test]
    async fn a_purge_reads_error_answers_sort_into_nothing_unanswered_and_refused() {
        // Issue #709: a purge walk's header read that the store
        // refuses has not judged its object, and its line names
        // the code: an AccessDenied is the credentials, an
        // InvalidObjectState an entry a lifecycle rule archived.
        // A 404 or a 416 holds nothing to purge, and a store
        // failure is no answer.
        let answer = |status: u16, body: &str| {
            reqwest::Response::from(
                axum::http::Response::builder()
                    .status(status)
                    .body(body.to_string())
                    .unwrap(),
            )
        };
        let archived = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error>\
                        <Code>InvalidObjectState</Code><Message>The operation is not \
                        valid for the object's storage class</Message></Error>";
        let read = HeaderRead::of_error(answer(403, archived), Instant::now()).await;
        assert!(
            matches!(&read, HeaderRead::Refused(status, Some(code))
                if *status == reqwest::StatusCode::FORBIDDEN && code == "InvalidObjectState"),
            "an archived entry's read is a refusal that names its code"
        );
        let read = HeaderRead::of_error(answer(400, ""), Instant::now()).await;
        assert!(matches!(read, HeaderRead::Refused(_, None)));
        for status in [404, 416] {
            let read = HeaderRead::of_error(answer(status, "<Error/>"), Instant::now()).await;
            assert!(matches!(read, HeaderRead::Nothing), "{status}");
        }
        for status in [500, 503, 429, 408] {
            let read = HeaderRead::of_error(answer(status, "<Error/>"), Instant::now()).await;
            assert!(matches!(read, HeaderRead::Failed), "{status}");
        }
    }

    /// Where the redirect test's store points: a listener that
    /// records the request line of each connection it accepts
    /// and answers `200` with nothing (issue #781).
    async fn start_elsewhere() -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/elsewhere", listener.local_addr().unwrap());
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut head = [0u8; 4096];
                let n = socket.read(&mut head).await.unwrap_or(0);
                let line = String::from_utf8_lossy(&head[..n])
                    .lines()
                    .next()
                    .unwrap_or("a connection with no request")
                    .to_string();
                recorded.lock().unwrap().push(line);
                let _ = socket
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .await;
            }
        });
        (base, seen)
    }

    #[tokio::test]
    async fn a_store_redirect_is_refused_and_never_followed() {
        // Issue #781: both clients followed a store's 3xx to
        // the host it named, with the session token, the date
        // and a write's body. A redirect is now the store's
        // answer: a refused lookup, a dropped write, a purge
        // walk left incomplete, an upload's failed check and
        // write, counted as the store's other refusals are,
        // and nothing reaches the host it names.
        let fake = start_fake().await;
        let (elsewhere, reached) = start_elsewhere().await;
        let mut creds = super::test_support::test_credentials();
        creds.session_token = Some("redirect-test-session-token".to_string());
        *fake.credentials.lock().unwrap() = Some(creds.clone());
        let s = S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(fake.endpoint.clone()),
            "us-east-1".into(),
            Some(creds.into()),
            test_entry_key(),
        )
        .unwrap();
        let cutoff = PurgeCutoff {
            start_ms: now_ms(),
            writer: 0xb,
        };
        s.put("k", &image(b"x", &["hero"])).await;
        assert!(s.put_raw("photo.png", b"x".to_vec()).await);
        assert!(
            s.put_tombstone(&tombstone_for("hero", cutoff, None), 1)
                .await
                .is_some()
        );
        assert_eq!(s.errors(), 0);
        let nowhere_else = |after: &str| {
            let reached = reached.lock().unwrap();
            assert!(
                reached.is_empty(),
                "{after} followed the store's redirect: {reached:?}"
            );
        };
        // The store redirects every request for an object.
        fake.fail.lock().unwrap().redirect_objects = Some(elsewhere.clone());
        assert!(s.get("k").await.is_none());
        nowhere_else("a lookup");
        assert_eq!((s.refused(Refused::Lookup), s.errors()), (1, 1));
        s.put("k2", &image(b"y", &[])).await;
        nowhere_else("a write");
        assert_eq!(s.errors(), 2);
        let hero = PurgeTarget::Group("hero".into());
        let swept = s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING).await;
        nowhere_else("a purge walk's read");
        assert!(!swept.complete, "the walk could not read the entry");
        assert_eq!((s.refused(Refused::Purge), s.errors()), (1, 3));
        assert!(
            s.put_tombstone(&tombstone_for("hero", cutoff, None), 2)
                .await
                .is_none()
        );
        nowhere_else("a tombstone write");
        assert_eq!(s.errors(), 4);
        let guard = PurgeGuard::new();
        assert!(!s.refresh_tombstones(&guard).await, "the tombstone read");
        nowhere_else("a tombstone read");
        assert_eq!(s.errors(), 5);
        // The upload bucket's store is this store too.
        assert!(!s.exists_raw("photo.png").await);
        nowhere_else("an existence check");
        assert_eq!((s.exists_checks_failed(), s.errors()), (1, 6));
        assert!(!s.put_raw("other.png", b"x".to_vec()).await);
        assert_eq!(
            s.put_raw_if_absent("new.png", b"x".to_vec()).await,
            RawPut::Failed
        );
        nowhere_else("an upload");
        // And every listing page: neither a walk nor a
        // tombstone refresh can list.
        fake.fail.lock().unwrap().redirect_listings = Some(elsewhere.clone());
        let swept = s.purge_whole(&hero, None, PurgeCutoff::EVERYTHING).await;
        assert!(!swept.complete, "the walk could not list");
        assert_eq!(s.errors(), 7);
        assert!(!s.refresh_tombstones(&guard).await, "the tombstone listing");
        assert_eq!(s.errors(), 8);
        nowhere_else("a listing");
        assert!(
            fake.requests
                .lock()
                .unwrap()
                .iter()
                .any(|line| line.starts_with("PUT cache/new.png")),
            "the store was asked"
        );
    }

    /// What [`start_hashless_store`] hands back: its endpoint,
    /// the receiver of each request's `x-amz-content-sha256` and
    /// body, and the sender that stops it.
    type HashlessStore = (
        String,
        std::sync::mpsc::Receiver<(Option<String>, Vec<u8>)>,
        tokio::sync::oneshot::Sender<()>,
    );

    /// A server that reads one request per connection, whole,
    /// and answers `200` without reading anything into it: on
    /// a runtime of its own on another thread, so nothing it
    /// does runs on the caller's.
    fn start_hashless_store() -> HashlessStore {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (seen_tx, seen_rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                ready_tx
                    .send(format!("http://{}", listener.local_addr().unwrap()))
                    .unwrap();
                let serve = async move {
                    while let Ok((mut socket, _)) = listener.accept().await {
                        let seen = seen_tx.clone();
                        tokio::spawn(async move {
                            let mut buf = Vec::new();
                            let mut chunk = vec![0u8; 64 * 1024];
                            let head_end = loop {
                                if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                                    break at + 4;
                                }
                                match socket.read(&mut chunk).await {
                                    Ok(0) | Err(_) => return,
                                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                                }
                            };
                            let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
                            let header = |name: &str| {
                                head.lines()
                                    .find_map(|l| l.strip_prefix(name)?.strip_prefix(':'))
                                    .map(|v| v.trim().to_string())
                            };
                            let length: usize = header("content-length")
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(0);
                            while buf.len() < head_end + length {
                                match socket.read(&mut chunk).await {
                                    Ok(0) | Err(_) => return,
                                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                                }
                            }
                            let _ = seen.send((
                                header("x-amz-content-sha256"),
                                buf[head_end..head_end + length].to_vec(),
                            ));
                            let _ = socket
                                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                                .await;
                        });
                    }
                };
                tokio::select! {
                    () = serve => {}
                    _ = stop_rx => {}
                }
            });
        });
        (ready_rx.recv().unwrap(), seen_rx, stop_tx)
    }

    #[tokio::test]
    async fn large_writes_hash_off_the_runtime_and_sign_the_same_digest() {
        // Issue #709: the whole-body SHA-256 of a write ran on a
        // runtime worker, and so did an entry's serialization,
        // which copies and MACs every byte: a large write held
        // the worker, and everything queued on it, for as long
        // as they took. They run on the blocking pool now, so a
        // ticker on this single-threaded runtime keeps ticking
        // through a large raw write and a large entry write, and
        // the store still receives the digest of the bytes it
        // gets.
        const BODY: usize = 16 * 1024 * 1024;
        let (endpoint, seen, stop) = start_hashless_store();
        let s = store(&endpoint);
        let raw = vec![0x5au8; BODY];
        let entry = image(&vec![0xa5u8; BODY], &[]);
        // What the hash alone costs here, in this build.
        let hashed = Instant::now();
        let _ = Sha256::digest(&raw);
        let hash_time = hashed.elapsed();
        let worst = Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO));
        let ticking = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let ticker = {
            let (worst, ticking) = (worst.clone(), ticking.clone());
            tokio::spawn(async move {
                let mut last = Instant::now();
                while ticking.load(std::sync::atomic::Ordering::SeqCst) {
                    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                    let now = Instant::now();
                    let mut worst = worst.lock().unwrap();
                    *worst = (*worst).max(now - last);
                    last = now;
                }
            })
        };
        assert!(s.put_raw("big.bin", raw.clone()).await);
        s.put("big", &entry).await;
        ticking.store(false, std::sync::atomic::Ordering::SeqCst);
        ticker.await.unwrap();
        let worst = *worst.lock().unwrap();
        assert!(
            worst < hash_time / 2,
            "the runtime stalled {worst:?}; one hash takes {hash_time:?}"
        );
        assert_eq!(s.errors(), 0);
        let (digest, body) = seen.recv().unwrap();
        assert_eq!(body, raw);
        assert_eq!(digest, Some(hex::encode(Sha256::digest(&raw))));
        let (digest, body) = seen.recv().unwrap();
        assert_eq!(body, serialize_entry("big", &entry, &test_entry_key()));
        assert_eq!(digest, Some(hex::encode(Sha256::digest(&body))));
        let _ = stop.send(());
    }

    /// A container credentials endpoint for the refresh tests:
    /// each fetch takes the next of `answers` (the last one
    /// repeats), a status and, for a 200, the credentials and
    /// how many seconds from now they expire. It records the
    /// `Authorization` header each fetch carried.
    async fn start_credentials_endpoint(
        answers: Vec<(u16, Option<(sigv4::Credentials, u64)>)>,
    ) -> (String, Arc<std::sync::Mutex<Vec<Option<String>>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/credentials", listener.local_addr().unwrap());
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = seen.clone();
        tokio::spawn(async move {
            let mut next = 0usize;
            while let Ok((mut socket, _)) = listener.accept().await {
                let (status, creds) = answers[next.min(answers.len() - 1)].clone();
                next += 1;
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                    }
                }
                let head = String::from_utf8_lossy(&buf).to_string();
                recorded.lock().unwrap().push(head.lines().find_map(|l| {
                    let (name, value) = l.split_once(':')?;
                    name.eq_ignore_ascii_case("authorization")
                        .then(|| value.trim().to_string())
                }));
                let body = match creds {
                    Some((creds, expires_in)) => {
                        let at = sigv4::amz_date(
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                                + expires_in,
                        );
                        format!(
                            "{{\"AccessKeyId\":\"{}\",\"SecretAccessKey\":\"{}\",\"Token\":\"{}\",\
                             \"Expiration\":\"{}-{}-{}T{}:{}:{}Z\"}}",
                            creds.access_key,
                            creds.secret_key,
                            creds.session_token.as_deref().unwrap_or_default(),
                            &at[0..4],
                            &at[4..6],
                            &at[6..8],
                            &at[9..11],
                            &at[11..13],
                            &at[13..15]
                        )
                    }
                    None => String::new(),
                };
                let answer = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(answer.as_bytes()).await;
            }
        });
        (url, seen)
    }

    fn role_creds(name: &str) -> sigv4::Credentials {
        sigv4::Credentials {
            access_key: format!("ASIA{name}"),
            secret_key: format!("{name}-secret/key+"),
            session_token: Some(format!("{name}-session-token")),
        }
    }

    #[tokio::test]
    async fn container_credentials_sign_and_refresh_before_they_expire() {
        // Issue #709: the keys were read once at startup, so
        // temporary ones expired until a restart, and then every
        // lookup was a silent miss. With no static keys, the
        // container endpoint's credentials sign, and a refresh
        // ahead of their expiry brings the next set before the
        // store stops honoring the first.
        use crate::credentials::{AwsCredentials, ContainerEndpoint, ContainerToken};
        let fake = start_fake().await;
        let (first, second) = (role_creds("FIRST"), role_creds("SECOND"));
        *fake.credentials.lock().unwrap() = Some(first.clone());
        let (url, seen) = start_credentials_endpoint(vec![
            (200, Some((first, 2))),
            (200, Some((second.clone(), 3600))),
        ])
        .await;
        let source = AwsCredentials::container_refreshing(
            ContainerEndpoint {
                url,
                token: Some(ContainerToken::Value(
                    "pod-identity-token".to_string().into(),
                )),
                var: "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            },
            std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(50),
        )
        .unwrap();
        let s = S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(fake.endpoint.clone()),
            "us-east-1".into(),
            Some(source.clone()),
            test_entry_key(),
        )
        .unwrap();
        let line = source.start().await.unwrap();
        assert!(
            line.starts_with("aws credentials: from the container credentials endpoint"),
            "{line}"
        );
        assert!(!line.contains("FIRST"), "{line}");
        s.put("k", &image(b"x", &[])).await;
        assert!(s.get("k").await.is_some());
        // A second before the first set expires, the refresh
        // brings the second, and the store honors only that.
        wait_for("the refresh", || {
            source
                .current()
                .is_some_and(|creds| creds.access_key == "ASIASECOND")
        })
        .await;
        *fake.credentials.lock().unwrap() = Some(second);
        assert!(s.get("k").await.is_some());
        assert_eq!(s.errors(), 0);
        assert_eq!(fake.refused(), Vec::<String>::new());
        let seen = seen.lock().unwrap().clone();
        assert!(seen.len() >= 2, "{seen:?}");
        assert!(
            seen.iter()
                .all(|auth| auth.as_deref() == Some("pod-identity-token")),
            "{seen:?}"
        );
    }

    #[tokio::test]
    async fn a_failed_first_fetch_is_retried_and_reads_its_token_file_each_time() {
        // Issue #709: an endpoint that fails at boot does not
        // stop it; the requests go unsigned and are refused,
        // counted, until a retry succeeds. The token file is
        // read at every fetch, as EKS rotates it.
        use crate::credentials::{AwsCredentials, ContainerEndpoint, ContainerToken};
        let fake = start_fake().await;
        let role = role_creds("ROLE");
        *fake.credentials.lock().unwrap() = Some(role.clone());
        let (url, seen) =
            start_credentials_endpoint(vec![(500, None), (200, Some((role, 3600)))]).await;
        let dir = crate::test_support::scratch_dir("s3-token-file");
        let token = dir.join("eks-pod-identity-token");
        std::fs::write(&token, "first-token\n").unwrap();
        let source = AwsCredentials::container_refreshing(
            ContainerEndpoint {
                url,
                token: Some(ContainerToken::File(token.clone())),
                var: "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            },
            crate::credentials::REFRESH_AHEAD,
            std::time::Duration::from_millis(500),
        )
        .unwrap();
        let s = S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(fake.endpoint.clone()),
            "us-east-1".into(),
            Some(source.clone()),
            test_entry_key(),
        )
        .unwrap();
        let line = source.start().await.unwrap();
        assert!(
            line.starts_with("warning: aws credentials: cannot fetch"),
            "{line}"
        );
        assert!(line.contains("500"), "{line}");
        assert!(s.get("k").await.is_none());
        assert_eq!(s.refused(Refused::Lookup), 1);
        std::fs::write(&token, "rotated-token").unwrap();
        wait_for("the retry", || source.current().is_some()).await;
        s.put("k", &image(b"x", &[])).await;
        assert!(s.get("k").await.is_some());
        assert_eq!(s.refused(Refused::Lookup), 1);
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen[0].as_deref(), Some("first-token"));
        assert_eq!(seen[1].as_deref(), Some("rotated-token"));
    }
}
