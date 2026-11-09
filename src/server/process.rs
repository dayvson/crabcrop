//! Processing scheduling (issue #345): the gates and
//! lanes a render job waits on, the singleflight maps
//! that coalesce concurrent requests for the same key
//! (image and meta flights), and the process runner
//! itself.
//!
//! server.rs re-exports the items the request path, main,
//! and the tests use.

use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode};

use super::AppState;
use crate::engine::EngineError;
use crate::url::ImageFormat;

/// In-flight miss coalescing (audit R2-09): the first
/// request to miss the cache for a key becomes its leader
/// and runs the fetch and the encode; concurrent requests
/// for the same key wait on the leader instead of each
/// fetching and encoding the same image, then re-check
/// the tiers. A result no tier stored (an error, a
/// fallback, a render with the caches off) reaches the
/// waiters through the flight's slot instead (issue
/// #407). The map holds one slot per key in flight; the
/// leader's guard removes the key and wakes the waiters
/// on drop, so every exit path (success, error,
/// fallback) releases them exactly once.
#[derive(Default)]
pub struct Inflight {
    leaders: std::sync::Mutex<std::collections::HashMap<String, Arc<FlightSlot>>>,
    /// Meta flights (issue #295): the same coalescing for
    /// meta requests, whose results are never cached, so
    /// the leader's document travels in a slot beside the
    /// wake channel instead of through the cache tiers.
    meta_leaders: std::sync::Mutex<std::collections::HashMap<String, Arc<MetaSlot>>>,
}

impl std::fmt::Debug for Inflight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inflight").finish_non_exhaustive()
    }
}

/// The caller's role for one cache key.
pub(in crate::server) enum Flight {
    /// This request runs the miss path and leaves its
    /// response in the slot; dropping the guard wakes the
    /// waiters.
    Leader(LeaderGuard, Arc<FlightSlot>),
    /// Another request is already running the miss path
    /// for this key; wait on the channel, then re-check
    /// the tiers, then the slot.
    Follower(tokio::sync::watch::Receiver<bool>, Arc<FlightSlot>),
}

/// The slot one image flight shares through (issue
/// #407): the leader fills `shared` before its guard
/// drops, and a woken follower whose tier lookup still
/// misses takes the response from it when its own
/// request is identical to the leader's.
pub(in crate::server) struct FlightSlot {
    sender: Arc<tokio::sync::watch::Sender<bool>>,
    pub(in crate::server) shared: std::sync::Mutex<Option<SharedFlight>>,
}

/// One image flight's response (issue #407). Only the
/// flight's followers ever see it: the slot dies with
/// the flight, so nothing is shared across time.
#[derive(Clone)]
pub(in crate::server) struct SharedFlight {
    /// Every input `serve` shaped the response from
    /// besides the conditional headers, as the miss path
    /// sees it (the raw path, the negotiated format, the
    /// hints; see `serve`): a follower takes the response
    /// only when its own request carries the same, since
    /// one cache key can serve spellings that answer
    /// differently (a sealed one answers errors with the
    /// uniform refusal, a preset one names its cache
    /// group).
    pub(in crate::server) request: String,
    /// The purge generation read before the leader's run
    /// began. A follower may have arrived after a purge
    /// that landed during the run (an overwritten upload,
    /// a purged URL), and such a request is owed a fresh
    /// render, so once a purge covering the key has been
    /// recorded since this reading no follower takes the
    /// response.
    pub(in crate::server) snapshot: u64,
    /// The response as the miss path produced it, before
    /// the leader's own conditional headers were applied.
    pub(in crate::server) response: SharedResponse,
    /// The ETag of the rendered bytes when the response
    /// serves a render, so every follower answers its own
    /// conditional request against them.
    pub(in crate::server) etag: Option<String>,
    /// Whether the response is the configured fallback
    /// image, for the follower's request record.
    pub(in crate::server) fallback: bool,
}

/// Removes the key from the in-flight map and wakes its
/// waiters when dropped, on every exit path alike.
pub(in crate::server) struct LeaderGuard {
    inflight: Arc<Inflight>,
    key: String,
    slot: Arc<FlightSlot>,
}

impl Drop for LeaderGuard {
    fn drop(&mut self) {
        self.inflight.remove(&self.key, &self.slot);
        let _ = self.slot.sender.send(true);
    }
}

impl Inflight {
    /// Claim leadership for `key`, or follow the request
    /// already leading it.
    pub(in crate::server) fn lead(self: &Arc<Self>, key: &str) -> Flight {
        let mut leaders = self.leaders.lock().unwrap();
        if let Some(slot) = leaders.get(key) {
            return Flight::Follower(slot.sender.subscribe(), slot.clone());
        }
        let (sender, _) = tokio::sync::watch::channel(false);
        let slot = Arc::new(FlightSlot {
            sender: Arc::new(sender),
            shared: std::sync::Mutex::new(None),
        });
        leaders.insert(key.to_string(), slot.clone());
        Flight::Leader(
            LeaderGuard {
                inflight: self.clone(),
                key: key.to_string(),
                slot: slot.clone(),
            },
            slot,
        )
    }

    /// Remove `key` if `slot` is still its leader (a
    /// waiter that already promoted itself must not lose
    /// its own claim to the previous leader's cleanup).
    fn remove(&self, key: &str, slot: &Arc<FlightSlot>) {
        let mut leaders = self.leaders.lock().unwrap();
        if leaders.get(key).is_some_and(|s| Arc::ptr_eq(s, slot)) {
            leaders.remove(key);
        }
    }

    /// Claim leadership of the meta flight for `key`, or
    /// follow the request already leading it.
    pub(in crate::server) fn lead_meta(self: &Arc<Self>, key: &str) -> MetaFlight {
        let mut leaders = self.meta_leaders.lock().unwrap();
        if let Some(slot) = leaders.get(key) {
            return MetaFlight::Follower(slot.sender.subscribe(), slot.clone());
        }
        let (sender, _) = tokio::sync::watch::channel(false);
        let slot = Arc::new(MetaSlot {
            sender: Arc::new(sender),
            shared: std::sync::Mutex::new(None),
        });
        leaders.insert(key.to_string(), slot.clone());
        MetaFlight::Leader(
            MetaLeaderGuard {
                inflight: self.clone(),
                key: key.to_string(),
                slot: slot.clone(),
            },
            slot,
        )
    }

    /// Remove `key` if `slot` is still its meta leader
    /// (the `remove` rule, for the meta map).
    fn remove_meta(&self, key: &str, slot: &Arc<MetaSlot>) {
        let mut leaders = self.meta_leaders.lock().unwrap();
        if leaders.get(key).is_some_and(|s| Arc::ptr_eq(s, slot)) {
            leaders.remove(key);
        }
    }
}

/// One response shared within a flight: the leader's
/// status, headers, and body, handed to the followers
/// that waited on it. Meta flights share every document,
/// errors included (issues #295, #407), and image flights
/// share what the tiers did not store; only within the
/// flight either way: the slot dies with the leader's
/// guard, so nothing is cached across time by sharing.
#[derive(Clone)]
pub(in crate::server) struct SharedResponse {
    pub(in crate::server) status: StatusCode,
    pub(in crate::server) headers: HeaderMap,
    pub(in crate::server) body: bytes::Bytes,
}

/// The slot one meta flight shares through: followers
/// subscribe to the wake channel, and the leader fills
/// `shared` before its guard drops, so a woken follower
/// either finds the document or learns the flight
/// produced nothing shareable.
pub(in crate::server) struct MetaSlot {
    sender: Arc<tokio::sync::watch::Sender<bool>>,
    pub(in crate::server) shared: std::sync::Mutex<Option<SharedResponse>>,
}

/// The caller's role for one meta flight key.
pub(in crate::server) enum MetaFlight {
    /// This request runs the miss path; dropping the
    /// guard wakes the waiters.
    Leader(MetaLeaderGuard, Arc<MetaSlot>),
    /// Another request is already running the miss path
    /// for this flight; wait on the channel, then read
    /// the slot.
    Follower(tokio::sync::watch::Receiver<bool>, Arc<MetaSlot>),
}

/// Removes the meta key from the in-flight map and wakes
/// its waiters when dropped, on every exit path alike
/// (the `LeaderGuard` pattern, for the meta map).
pub(in crate::server) struct MetaLeaderGuard {
    inflight: Arc<Inflight>,
    key: String,
    slot: Arc<MetaSlot>,
}

impl Drop for MetaLeaderGuard {
    fn drop(&mut self) {
        self.inflight.remove_meta(&self.key, &self.slot);
        let _ = self.slot.sender.send(true);
    }
}

/// A bounded concurrency gate with a known capacity: a
/// semaphore plus the permit count it was built with, so
/// `/metrics` can report lane occupancy as in-use against
/// capacity instead of only the permits left.
#[derive(Debug, Clone)]
pub struct Gate {
    sem: Arc<tokio::sync::Semaphore>,
    permits: usize,
}

impl Gate {
    pub fn new(permits: usize) -> Self {
        Self {
            sem: Arc::new(tokio::sync::Semaphore::new(permits)),
            permits,
        }
    }

    /// The permit count the gate was built with.
    pub fn capacity(&self) -> usize {
        self.permits
    }

    /// Permits currently free.
    pub fn available_permits(&self) -> usize {
        self.sem.available_permits()
    }

    /// Permits currently held: capacity minus free.
    pub fn in_use(&self) -> usize {
        self.permits.saturating_sub(self.available_permits())
    }

    /// Take one permit, owned so it can move across an
    /// await or into a detached guard.
    pub fn acquire_owned(
        self: Arc<Self>,
    ) -> impl std::future::Future<
        Output = Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::AcquireError>,
    > {
        self.sem.clone().acquire_owned()
    }
}

/// An owned bundle for one engine run: everything
/// [`crate::engine::ProcessContext`] borrows, owned so the work
/// can move to a blocking thread (audit E-5).
#[derive(Clone)]
pub(in crate::server) struct ProcessJob {
    pub(in crate::server) save_data: bool,
    pub(in crate::server) preserve_metadata: bool,
    pub(in crate::server) ops: Vec<crate::url::Operation>,
    pub(in crate::server) preferred: Option<ImageFormat>,
    /// Watermark images as (URL, fetched bytes) pairs: the
    /// decode happens inside the gated blocking job, not in
    /// the async handler (audit R2-16). Shared, so cloning
    /// the job for a run never copies them (issue #403).
    pub(in crate::server) watermark_sources: Arc<WatermarkSources>,
    pub(in crate::server) placeholders: bool,
    pub(in crate::server) mask_provider: Option<Arc<dyn crate::ai::MaskProvider>>,
    pub(in crate::server) upscale_provider: Option<Arc<dyn crate::ai::UpscaleProvider>>,
    pub(in crate::server) inpaint_provider: Option<Arc<dyn crate::ai::InpaintProvider>>,
    pub(in crate::server) segment_provider: Option<Arc<dyn crate::ai::SegmentProvider>>,
    pub(in crate::server) face_provider: Option<Arc<dyn crate::ai::FaceProvider>>,
    pub(in crate::server) detect_provider: Option<Arc<dyn crate::ai::DetectionProvider>>,
    pub(in crate::server) declared_detect_providers: Vec<Arc<dyn crate::ai::DetectionProvider>>,
    pub(in crate::server) focal_provider: Option<Arc<dyn crate::ai::FocalProvider>>,
    pub(in crate::server) text_renderer: Option<Arc<dyn crate::text::TextRenderer>>,
    pub(in crate::server) video_encoder: Option<Arc<crate::video::VideoEncoder>>,
    pub(in crate::server) frame_extractor: Option<Arc<crate::frames::VideoFrameExtractor>>,
    pub(in crate::server) pdf_renderer: Option<Arc<crate::pdf::PdfRenderer>>,
    /// C2PA output signer (issue #173) when the operator
    /// configured one.
    pub(in crate::server) provenance_signer: Option<Arc<crate::provenance::ProvenanceSigner>>,
    /// Whether this request's operations include an AI
    /// edit (removebg, studio, remove_region, upscale,
    /// class edits, face redactions), recorded on the
    /// output manifest's edit action.
    pub(in crate::server) ai_edited: bool,
    /// Whether the caller wants provenance facts computed
    /// for the response (meta requests).
    pub(in crate::server) want_provenance: bool,
    pub(in crate::server) dpr: Option<f32>,
    pub(in crate::server) ssim_target: f64,
    pub(in crate::server) brand_color: Option<[u8; 3]>,
    pub(in crate::server) avif_speed: u8,
    pub(in crate::server) jxl_speed: jixel::Speed,
}

/// A request's watermark images, with their share of the
/// source buffer budget (issue #403). A run and its
/// fallback retry share one copy; the share returns to the
/// budget when the last holder drops it, after the images
/// (fields drop in declaration order), so the budget never
/// reports less than what is still buffered.
#[derive(Default)]
pub(in crate::server) struct WatermarkSources {
    pub(in crate::server) images: Vec<(String, Vec<u8>)>,
    /// Held only to be dropped after `images`.
    #[allow(dead_code)]
    pub(in crate::server) share: Option<crate::loader::SourceCharge>,
}

/// Why a gated engine run produced no image.
#[derive(Debug)]
pub(in crate::server) enum ProcessFailure {
    /// The engine's own error; fallback logic applies.
    Engine(EngineError),
    /// The process gate stayed saturated past its wait:
    /// the request is shed with a 503.
    Busy,
    /// The engine panicked. The panic is contained at this
    /// seam (audit E-8) and surfaces as a 500.
    Panicked,
    /// The job ran past its processing deadline plus the
    /// grace bound (issue #322): the runner stopped
    /// waiting and the request is shed with a 504. The
    /// blocking job itself is detached and keeps every
    /// permit `run_process` moved into it, the lane
    /// permit and the request's AI, video, and animation
    /// permits, until it finishes (issue #357): the gates
    /// count the work still draining, so a follow-up
    /// request waits or sheds at the gate instead of
    /// stacking onto busy capacity. The detached job
    /// aborts at the engine's next deadline check; a job
    /// stuck inside one opaque call runs only until that
    /// call returns.
    TimedOut,
}

/// How long a cache miss may wait for a processing slot
/// before the server sheds it with a 503.
pub(in crate::server) const PROCESS_GATE_WAIT: std::time::Duration =
    std::time::Duration::from_secs(5);

/// How long a cache miss, or an upload, may wait in total
/// for source buffer budget before the server sheds it with
/// a 503 (issue #403): the stills gate's wait, since every
/// miss, still or slow, charges its bytes before the gates,
/// and a share frees as soon as a render ends.
pub(in crate::server) const SOURCE_BUDGET_WAIT: std::time::Duration = PROCESS_GATE_WAIT;

/// Which processing lane an engine run takes its permit
/// from (issue #324). The lanes are fully separate: a run
/// holds a permit from exactly one of them, so saturation
/// of one cannot reduce the other's capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::server) enum Lane {
    /// The stills lane (the process gate): ordinary still
    /// renders, the common and fast case.
    Process,
    /// The slow lane: long-running work, animated WebP
    /// encodes and the external-tool work behind the
    /// video gate (video encodes, frame extraction, page
    /// renders), whose runs last seconds to minutes.
    Slow,
}

/// The slow lane's default size (issue #324), from
/// `CRABCROP_SLOW_LANE_PERMITS`: one long-running encode
/// at a time, sane for a 2 vCPU host, where one slow
/// encode leaves a core and the whole stills lane free.
pub const DEFAULT_SLOW_LANE_PERMITS: usize = 1;

/// How long a slow-lane run may wait for a slow slot
/// before the server sheds it with a 503 (issue #324).
/// The animation and video gates' wait: long enough to
/// absorb a hiccup, far shorter than queueing behind a
/// minutes-long encode.
pub(in crate::server) const SLOW_LANE_WAIT: std::time::Duration =
    std::time::Duration::from_secs(10);

/// Whether this miss belongs in the slow lane (issue
/// #324): it will encode an animated WebP (`animated_webp`,
/// the animation gate's own predicate, decided once per
/// miss so the gate and the lane always agree), or it is
/// external-tool work behind the video gate, a video
/// encode, a `frame:` extraction, or a `page:` render, all
/// subprocess work that runs for seconds and is
/// unpredictable in the same way. Everything else, still
/// renders in particular, stays in the process lane. AI
/// transforms stay there too: they are still renders
/// behind the AI gate, not encodes, and the AI gate
/// already bounds them separately. Animated GIF output
/// stays there as well: its encoder is chunked per frame
/// with deadline checks inside the loop (issue #322), not
/// one opaque long call like the animated WebP encoder.
pub(in crate::server) fn wants_slow_lane(
    animated_webp: bool,
    wants_video: bool,
    wants_frame: bool,
    wants_page: bool,
) -> bool {
    animated_webp || wants_video || wants_frame || wants_page
}

/// The grace bound past the processing deadline (issue
/// #322): how long the runner keeps waiting for a job
/// whose deadline has passed before it answers 504 and
/// detaches the job; the job's gates stay with it until
/// it finishes (issue #357). The engine checks the deadline between
/// stages and inside its chunked encode loops (per frame
/// for animated GIF and WebP, per attempt for the
/// quality ladders), so a checkpointed job aborts on its
/// own within this window and answers with the engine's
/// own Timeout. A job still running when the grace runs
/// out is inside a single opaque encoder call (one still
/// encode, libwebp's animated `try_encode`, one AI
/// inference, the external video encoder), which no
/// checkpoint can interrupt: detaching is what bounds
/// its hold on the server, and it aborts at the first
/// deadline check after that call returns. The detached
/// job keeps its permits deliberately (issue #357): the
/// deadline checks bound the hold (the deadline plus the
/// longest single opaque call).
const DEADLINE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// How long an AI-backed request may wait for an inference
/// slot before the server sheds it with a 503 (audit PL-14).
pub(in crate::server) const AI_GATE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long a video request may wait for an encode slot
/// before the server sheds it with a 503. Encodes run for
/// seconds behind a 2-permit gate; the wait is the AI
/// gate's, and a shed request can retry into a drained
/// queue.
pub(in crate::server) const VIDEO_GATE_WAIT: std::time::Duration =
    std::time::Duration::from_secs(10);

/// How long an animated WebP request may wait for the
/// single animation encode slot before the server sheds
/// it with a 503 (issue #231). The AI and video gates'
/// wait: long enough to absorb a hiccup, far shorter
/// than queueing behind a minutes-long encode.
pub(in crate::server) const ANIM_GATE_WAIT: std::time::Duration =
    std::time::Duration::from_secs(10);

/// Whether this miss will encode an animated WebP and so
/// must take the animation gate (issue #231) and the slow
/// lane (issue #324): the source is animated, an animated
/// WebP or an animated GIF (probes on the loaded bytes, no
/// decode), and its output resolves to WebP through the
/// engine's own resolution, the explicit `format(...)` or
/// else the format negotiated from `Accept` (`preferred`).
/// An animated GIF rendered to WebP keeps its frames and
/// runs the same libwebp animated encoder as an animated
/// WebP source (issue #370), so this predicate covers both
/// sources (issue #405). The GIF
/// probe walks the first frame's compressed bytes, so it
/// runs only when the output would resolve to WebP.
pub(in crate::server) fn wants_animation_gate(
    bytes: &[u8],
    directives: &crate::url::RequestDirectives,
    preferred: Option<ImageFormat>,
) -> bool {
    let to_webp = |source| {
        crate::engine::animated_output_format(source, directives.explicit_format, preferred)
            == ImageFormat::WebP
    };
    if crate::engine::is_animated_webp(bytes) {
        return to_webp(ImageFormat::WebP);
    }
    to_webp(ImageFormat::Gif) && crate::engine::is_animated_gif(bytes)
}

/// The gate permits a request acquired before its run,
/// beyond the lane permit `run_process` takes itself
/// (issue #357): the AI, video, and animation permits.
/// `run_process` moves them into the blocking job, so
/// each is held until the job actually finishes, even
/// when the runner has already shed the request.
#[derive(Default)]
pub(in crate::server) struct JobPermits {
    pub(in crate::server) ai: Option<tokio::sync::OwnedSemaphorePermit>,
    pub(in crate::server) video: Option<tokio::sync::OwnedSemaphorePermit>,
    pub(in crate::server) anim: Option<tokio::sync::OwnedSemaphorePermit>,
}

/// The source bytes a run takes over (issue #403), with
/// their share of the source buffer budget: the blocking
/// job owns both, so the bytes are never copied for it,
/// and the share returns to the budget only after the job
/// has released the bytes, even when the runner has
/// already shed the request.
pub(in crate::server) struct JobSource {
    pub(in crate::server) bytes: Vec<u8>,
    pub(in crate::server) share: Option<crate::loader::SourceCharge>,
}

/// Tests lend their fixture bytes by reference; the copy
/// is theirs, and the run holds no budget share.
#[cfg(test)]
impl From<&Vec<u8>> for JobSource {
    fn from(bytes: &Vec<u8>) -> Self {
        JobSource {
            bytes: bytes.clone(),
            share: None,
        }
    }
}

/// Run the engine under its lane's gate (issue #324) on
/// a blocking thread, so CPU-bound work neither stalls
/// the async workers nor multiplies unboundedly under a
/// burst of cache misses. Cache hits never reach this
/// function. The run takes the source by value (issue
/// #403): the blocking job owns the bytes, with their
/// budget share, instead of copying them.
pub(in crate::server) async fn run_process(
    state: &AppState,
    job: &ProcessJob,
    source: impl Into<JobSource>,
    lane: Lane,
    deadline: std::time::Instant,
    permits: JobPermits,
) -> Result<crate::engine::ProcessedImage, ProcessFailure> {
    let source: JobSource = source.into();
    // The run takes its permit from its lane's gate only
    // (issue #324): a slow-lane run never touches a
    // stills permit, and a still never waits behind the
    // slow lane. Overflow in either lane is the same
    // honest Busy shed, after that lane's wait.
    let (gate, wait) = match lane {
        Lane::Process => (state.process_gate.clone(), PROCESS_GATE_WAIT),
        Lane::Slow => (state.slow_gate.clone(), SLOW_LANE_WAIT),
    };
    let permit = match tokio::time::timeout(wait, gate.acquire_owned()).await {
        Ok(Ok(permit)) => permit,
        _ => return Err(ProcessFailure::Busy),
    };
    let job = job.clone();
    let engine = state.engine;
    // The processing deadline (issue #260) is the
    // caller's: serve_miss stamps it when the process
    // phase starts, so the gate waits above share the
    // request's one process budget instead of the run
    // starting a fresh timeout after them. The engine
    // checks it between stages (operations, frames) and
    // inside its chunked encode loops and aborts chained
    // work promptly. The runner's own wait is bounded by
    // the deadline plus the grace bound (issue #322).
    let handle = tokio::task::spawn_blocking(move || {
        // Every permit the run holds is accounted the
        // same way (issue #357): the lane permit and the
        // AI, video, and animation permits all live here,
        // in the blocking closure, and drop only when the
        // job actually finishes. A run the runner has
        // already shed therefore keeps its slots while it
        // drains, so the gates count work still executing
        // and not only requests still waiting, and new
        // work sheds at a gate instead of stacking onto
        // capacity that is still busy. The hold a shed
        // run leaves behind is bounded: the job carries
        // the deadline, the engine checks it between
        // stages and inside its chunked loops, and a job
        // inside one opaque call (a still encode,
        // libwebp's animated encode, an inference, the
        // external video encoder) unwinds at the first
        // check after that call returns, so a permit is
        // held for the deadline plus the longest single
        // call, never open-endedly.
        let _lane_permit = permit;
        let _ai_permit = permits.ai;
        let _video_permit = permits.video;
        let _anim_permit = permits.anim;
        let JobSource { bytes, share } = source;
        let declared_refs: Vec<&dyn crate::ai::DetectionProvider> = job
            .declared_detect_providers
            .iter()
            .map(|p| p.as_ref())
            .collect();
        // Watermark images decode here, inside the gated
        // blocking job, rather than inline in the async
        // handler: the decode is CPU work a burst of
        // watermarked requests must not run ungated on the
        // runtime workers, and a decoder panic on crafted
        // watermark bytes is contained by the same
        // catch_unwind as the engine run (audit R2-16).
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut watermarks = crate::filters::WatermarkImages::new();
            for (url, wm_bytes) in &job.watermark_sources.images {
                // An AVIF watermark decodes in the capped worker
                // under the request deadline, like the source
                // (issue #522).
                let img = crate::engine::decode_source_image(wm_bytes, Some(deadline))?;
                watermarks.insert(url.clone(), img);
            }
            let ctx = crate::engine::ProcessContext {
                preferred: job.preferred,
                watermarks: Some(&watermarks),
                placeholders: job.placeholders,
                mask_provider: job.mask_provider.as_deref(),
                upscale_provider: job.upscale_provider.as_deref(),
                inpaint_provider: job.inpaint_provider.as_deref(),
                segment_provider: job.segment_provider.as_deref(),
                face_provider: job.face_provider.as_deref(),
                detect_provider: job.detect_provider.as_deref(),
                declared_detect_providers: &declared_refs,
                focal_provider: job.focal_provider.as_deref(),
                video_encoder: job.video_encoder.as_deref(),
                frame_extractor: job.frame_extractor.as_deref(),
                pdf_renderer: job.pdf_renderer.as_deref(),
                save_data: job.save_data,
                preserve_metadata: job.preserve_metadata,
                text_renderer: job.text_renderer.as_deref(),
                dpr: job.dpr,
                ssim_target: job.ssim_target,
                brand_color: job.brand_color,
                avif_speed: job.avif_speed,
                jxl_speed: job.jxl_speed,
                deadline: Some(deadline),
            };
            let mut processed = engine.process_preferred(&bytes, &job.ops, ctx)?;
            apply_provenance(&job, &bytes, &mut processed)?;
            Ok(processed)
        })) {
            Ok(Ok(processed)) => Ok(processed),
            Ok(Err(e)) => Err(ProcessFailure::Engine(e)),
            Err(_) => Err(ProcessFailure::Panicked),
        };
        // The source goes before its budget share does
        // (issue #403), so the budget never reports less
        // than what this job still holds; the watermarks
        // carry their own share (see `WatermarkSources`).
        drop(bytes);
        drop(share);
        result
    });
    // Past the deadline plus the grace bound the runner
    // stops waiting and sheds the request with a 504
    // (issue #322). The detached job keeps every permit
    // it holds until it finishes (issue #357), so
    // a follow-up request waits at the gate for the
    // bounded drain and sheds there if the drain outlasts
    // its gate wait, rather than starting work the
    // server has no free capacity for. The detached
    // job's result is dropped; it aborts at the engine's
    // next deadline check, or, when it is stuck inside a
    // single opaque encoder call, as soon as that call
    // returns and the next check runs.
    match tokio::time::timeout_at(
        tokio::time::Instant::from_std(deadline + DEADLINE_GRACE),
        handle,
    )
    .await
    {
        Ok(joined) => joined.unwrap_or(Err(ProcessFailure::Panicked)),
        Err(_) => Err(ProcessFailure::TimedOut),
    }
}

/// The C2PA step (issue #173), run inside the gated
/// process job: verify what the source carries and, when
/// a signer is configured, sign the output. Byte
/// preservation of a source manifest is impossible (the
/// pipeline drops the segments it lives in), so the
/// source's manifest rides into the output's manifest
/// store as a parent ingredient instead, keeping the
/// provenance chain intact. Signing covers still JPEG,
/// PNG, WebP, and AVIF outputs; GIF, JPEG XL, video, and
/// animated WebP outputs are served unsigned. A signing
/// failure fails the request: serving unmarked output
/// from a server configured to mark would be a silent
/// policy break. A failed timestamp authority gets its own
/// error, so the client hears an outage worth retrying
/// rather than a broken encoder (issue #467).
pub(in crate::server) fn apply_provenance(
    job: &ProcessJob,
    source_bytes: &[u8],
    processed: &mut crate::engine::ProcessedImage,
) -> Result<(), EngineError> {
    let signer = &job.provenance_signer;
    if signer.is_none() && !job.want_provenance {
        return Ok(());
    }
    let source_format = crate::provenance::sniff_format(source_bytes);
    // The source summary is a meta-only fact: reading it
    // validates the source's whole manifest store, work
    // an image response has no use for.
    let source = if job.want_provenance {
        crate::provenance::read_manifest(source_bytes, source_format)
    } else {
        None
    };
    let mut signed = false;
    if let Some(signer) = signer {
        let mime = crate::provenance::signable_mime(processed.format);
        let animated_webp = processed.format == ImageFormat::WebP
            && crate::provenance::webp_is_animated(&processed.bytes);
        if let (Some(mime), false) = (mime, animated_webp) {
            // The source asset rides as a parent
            // ingredient whenever it is an image (any
            // format the engine's sniff names, which is the
            // sniff it decodes by), manifest or not: the
            // ingredient names its format, and carries its
            // manifest store when it has one. Video and PDF
            // sources are the only others: the output is a
            // frame or page rendered from them, and starts
            // a fresh chain (see provenance::sign).
            let ingredient =
                crate::provenance::is_image(source_format).then_some((source_bytes, source_format));
            let request = crate::provenance::SignRequest {
                ingredient,
                ai_edited: job.ai_edited,
            };
            match signer.sign(&processed.bytes, mime, &request) {
                Ok(bytes) => {
                    processed.bytes = bytes;
                    signed = true;
                }
                Err(crate::provenance::SignError::Authority(cause)) => {
                    return Err(EngineError::ProvenanceUnavailable(cause));
                }
                Err(crate::provenance::SignError::Failed(cause)) => {
                    return Err(EngineError::Encode(format!("C2PA signing failed: {cause}")));
                }
            }
        }
    }
    if job.want_provenance {
        let output = if signed {
            crate::provenance::read_manifest(
                &processed.bytes,
                crate::provenance::sniff_format(&processed.bytes),
            )
        } else {
            None
        };
        processed.provenance = Some(crate::provenance::ProvenanceFacts {
            source,
            output,
            signing_configured: signer.is_some(),
        });
    }
    Ok(())
}
