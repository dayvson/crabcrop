use std::sync::Arc;
use std::time::Instant;

use axum::http::{StatusCode, header};
use axum::response::Response;

use super::admission::width_hint_ops;
use super::request::OwedRecord;
use super::response::{
    attach_cache_headers, busy_response, engine_error_response, error_response,
    load_error_response, respond, sealed_or,
};
use super::tiers::{DiskHandoff, populate_tiers};
use super::{
    AI_GATE_WAIT, ANIM_GATE_WAIT, AppState, Gate, JobPermits, JobSource, Lane, MissProgress,
    ProcessFailure, ProcessJob, SOURCE_BUDGET_WAIT, VIDEO_GATE_WAIT, WatermarkSources,
    meta_document, run_process, wants_animation_gate, wants_slow_lane,
};
use crate::cache::{CachedImage, etag_for_bytes};
use crate::engine::EngineError;
use crate::loader::LoadError;
use crate::observe::{FailStage, RequestRecord};
use crate::url::ImageFormat;

#[cfg(doc)]
use super::admission::Negotiated;

/// The operator-configured fallback image: processed and
/// served in place of a source that cannot be loaded or
/// decoded, so a dead origin degrades to a placeholder instead
/// of a broken image.
#[derive(Debug, Clone)]
pub enum Fallback {
    /// A local file, read once at boot.
    Bytes(Arc<Vec<u8>>),
    /// A URL, fetched per fallback event through the loader
    /// (and its bounds and source policy).
    Url(String),
}

/// The fallback's bytes, when one is configured and reachable.
/// A URL fallback that itself fails to load yields `None`, and
/// the caller answers with the original source error. The copy
/// is charged to the request's source budget share like a
/// source (issue #403); no budget within the wait is `None` too.
async fn fallback_source_bytes(
    state: &AppState,
    charge: &crate::loader::SourceCharge,
) -> Option<Vec<u8>> {
    match &state.fallback {
        Some(Fallback::Bytes(bytes)) => {
            charge.grow_to(charge.bytes() + bytes.len()).await.ok()?;
            Some(bytes.as_ref().clone())
        }
        Some(Fallback::Url(url)) => state
            .loader
            .load_media_charged(url, crate::loader::MediaAdmission::default(), charge)
            .await
            .ok(),
        None => None,
    }
}

/// Everything the cache-miss path needs, bundled so the
/// singleflight loop in `run_flight` can hand it to
/// whichever request leads the miss.
pub(in crate::server) struct MissWork {
    pub(in crate::server) ops: Vec<crate::url::Operation>,
    pub(in crate::server) source_url: String,
    pub(in crate::server) directives: crate::url::RequestDirectives,
    pub(in crate::server) preferred: Option<ImageFormat>,
    pub(in crate::server) meta: bool,
    pub(in crate::server) debug: bool,
    pub(in crate::server) placeholders: bool,
    pub(in crate::server) hinted_width: Option<u32>,
    pub(in crate::server) dpr_hint: Option<f32>,
    /// Whether Save-Data steps this render's quality:
    /// [`Negotiated::save_data_applies`], the rule the cache
    /// key and the private marking follow, not the raw
    /// header.
    pub(in crate::server) save_data: bool,
    pub(in crate::server) accept: Option<String>,
    pub(in crate::server) groups: Vec<String>,
    pub(in crate::server) cache_key: String,
    pub(in crate::server) max_age: Option<u32>,
    /// Whether the request arrived in the sealed spelling
    /// (issue #255). Meta for it redacts every operation and
    /// source text, and its image responses carry no
    /// Cache-Group (issue #454). The flag describes this
    /// request, not the render: meta bypasses the cache in
    /// both directions, so a plain meta request on the same
    /// render still reports in full, and a plain hit on an
    /// entry first served sealed still names its groups.
    pub(in crate::server) sealed: bool,
    /// How far this miss has got (see [`MissProgress`]);
    /// its render jobs carry clones.
    pub(in crate::server) progress: MissProgress,
    /// Where a flight leader's render leaves its disk write,
    /// to finish once its followers are woken (issue #572).
    /// `None` everywhere else.
    pub(in crate::server) disk_handoff: Option<DiskHandoff>,
}

/// Whether a request runs inference and so takes an AI
/// gate permit (issue #379): every AI directive counts,
/// except a smart crop, which runs a model only when a
/// face or focal provider participates in it. Without
/// either provider the crop's focal point is the built-in
/// saliency statistic, no inference runs, and gating the
/// request would let it consume permits and shed behind
/// real AI work for nothing.
pub(in crate::server) fn ai_gate_applies(
    directives: &crate::url::RequestDirectives,
    face_provider: bool,
    focal_provider: bool,
) -> bool {
    directives.uses_ai_beyond_smart_crop()
        || (directives.smart_crop && (face_provider || focal_provider))
}

/// The gates a miss's engine run takes beyond its lane
/// permit (issue #357).
#[derive(Clone, Copy)]
struct GateNeeds {
    /// The AI gate: the request runs inference (see
    /// [`ai_gate_applies`]).
    ai: bool,
    /// The video gate: a video encode, a `frame:`
    /// extraction, or a `page:` render.
    video: bool,
    /// The animation gate: an animated WebP encode (see
    /// [`wants_animation_gate`]).
    anim: bool,
}

/// The gate a run found still full when its wait ran out.
#[derive(Clone, Copy)]
enum Busy {
    Ai,
    Video,
    Anim,
}

impl Busy {
    /// The first run's 503 for the full gate. A fallback
    /// retry that meets one leaves the original engine
    /// error standing instead.
    fn response(self, accept: Option<&str>) -> Response {
        let message = match self {
            Busy::Ai => "server busy: AI processing queue is full",
            Busy::Video => "server busy: video processing queue is full",
            Busy::Anim => "server busy: animation encoding queue is full",
        };
        busy_response(message, 5, accept)
    }
}

/// One permit from `gate`, waiting at most `wait` for it.
async fn acquire_gate(
    gate: &Arc<Gate>,
    wait: std::time::Duration,
) -> Option<tokio::sync::OwnedSemaphorePermit> {
    match tokio::time::timeout(wait, gate.clone().acquire_owned()).await {
        Ok(Ok(permit)) => Some(permit),
        _ => None,
    }
}

/// Take the permits a miss's run needs beyond its lane
/// permit, AI then video then animation: the one
/// acquisition the first run and the fallback retry share
/// (issue #479). Each wait is bounded (audit PL-14): a
/// burst sheds with a 503 instead of parking unbounded
/// tasks on a gate. The permits travel with the process
/// job, and `run_process` holds them until the job
/// actually finishes (issue #357). A gate still full when
/// its wait runs out is the [`Busy`] it names, and the
/// permits already taken drop with it.
async fn acquire_job_permits(state: &AppState, needs: GateNeeds) -> Result<JobPermits, Busy> {
    let mut permits = JobPermits::default();
    // AI inference is gated process-wide: a burst of removebg
    // requests must not starve the transform pipeline.
    if needs.ai {
        permits.ai = Some(
            acquire_gate(&state.ai_gate, AI_GATE_WAIT)
                .await
                .ok_or(Busy::Ai)?,
        );
    }
    // Video work sits behind its own gate, on top of the
    // slow lane inside `run_process` (issue #324): the
    // external encoder is the server's most expensive and
    // least predictable work (seconds per job, a
    // subprocess), so a burst of video requests sheds here
    // instead of queueing unboundedly behind the transform
    // pipeline, and its lane permit comes from the slow
    // lane, never from the stills pool.
    // Frame extraction (issue #214) takes the same gate:
    // it runs the same class of subprocess. Page
    // rendering (issue #215) takes it too: it is the
    // external-tool gate in everything but name, and a
    // second gate for the same class of work would only
    // split the protection.
    if needs.video {
        permits.video = Some(
            acquire_gate(&state.video_gate, VIDEO_GATE_WAIT)
                .await
                .ok_or(Busy::Video)?,
        );
    }
    // Animated WebP encodes sit behind a single-permit
    // gate of their own (issue #231): one encode can run
    // for minutes, so without this gate duplicate or
    // distinct animated renders stack encodes until the
    // stills pool is drained. Behind the gate, a second
    // animated request sheds here with an explanation
    // instead of stacking, and the encode's lane permit
    // comes from the slow lane (issue #324), so the still
    // pool keeps every permit throughout.
    if needs.anim {
        permits.anim = Some(
            acquire_gate(&state.anim_gate, ANIM_GATE_WAIT)
                .await
                .ok_or(Busy::Anim)?,
        );
    }
    Ok(permits)
}

/// How a miss's engine runs go, decided once per miss so
/// the first run and a fallback retry agree: the lane each
/// takes its permit from (issue #324), the gates each takes
/// beyond it, and when the process phase began, which
/// starts the request's one deadline.
#[derive(Clone, Copy)]
struct RunPlan {
    lane: Lane,
    needs: GateNeeds,
    start: Instant,
}

/// What a miss rendered.
pub(in crate::server) struct Rendered {
    processed: crate::engine::ProcessedImage,
    /// Whether it was rendered from the configured
    /// fallback rather than the source.
    used_fallback: bool,
}

/// The cache-miss path: load the source, resolve
/// watermarks, run the engine under the gates, populate
/// the tiers, and respond (a meta document when asked).
/// Runs at most once per key at a time, led by one
/// request while its duplicates wait (see `run_flight`).
/// The third return value is the ETag of the rendered
/// bytes when the response serves a render, so the
/// caller can answer conditional requests against the
/// bytes themselves; it is `None` for a meta document
/// and for every error.
///
/// Every stage that can fail runs in [`render_miss`], and
/// its failure is answered here, the one place on the
/// miss path that knows the request's spelling (issue
/// #479).
pub(in crate::server) async fn serve_miss(
    state: &AppState,
    work: MissWork,
    mut record: OwedRecord,
) -> (Response, OwedRecord, Option<String>) {
    // The purge generation this render starts under
    // (issue #350), taken before the load: a purge that
    // lands while the render runs must not be undone by
    // the population writes at its end.
    let write_snapshot = state.purge_guard.snapshot();
    // Its write stamp (issues #446, #461), taken after the
    // snapshot: a purge that begins after this point covers
    // the copies this render writes (it may read the old
    // source), and one that began before it does not, so
    // they outlive that purge's walk.
    let write_stamp = state.purge_guard.stamp();
    let rendered = render_miss(state, &work, &mut record).await;
    let MissWork {
        meta,
        debug,
        accept,
        groups,
        cache_key,
        max_age,
        sealed,
        disk_handoff,
        ..
    } = work;
    let Rendered {
        processed,
        used_fallback,
    } = match rendered {
        Ok(rendered) => rendered,
        // Once a token has opened, every failure answers
        // the uniform refusal (issues #255, #259); a plain
        // request gets the failure's own answer.
        Err(refusal) => return (sealed_or(sealed, *refusal), record, None),
    };
    let accept = accept.as_deref();
    let cache_status = "crabcrop; fwd=uri-miss";
    // The validator for this render is a digest of its
    // bytes, computed once here: it is stored with the
    // cache entry and returned to the caller, so a hit and
    // a fresh render answer conditional requests against
    // the same value, and a re-render under the same key
    // whose bytes differ (the source changed, the old entry
    // was purged) answers with a different one instead of
    // the key's old validator.
    let render_etag = etag_for_bytes(&processed.bytes);

    // Only successful responses are cached; errors never are.
    // Meta responses are not image entries and are not cached.
    // Fallback responses are not cached either: the placeholder
    // must never be stored as if it were the real image, so the
    // origin gets retried on the next request. Debug responses
    // are not cached: the diagnostic is not the transform
    // result its key would claim.
    if !meta && !used_fallback && !debug {
        let entry = CachedImage {
            bytes: bytes::Bytes::from(processed.bytes.clone()),
            format: processed.format,
            quality: processed.quality,
            groups: groups.clone(),
            etag: render_etag.clone(),
            written: Some(write_stamp),
        };
        populate_tiers(
            state,
            write_snapshot,
            &cache_key,
            entry,
            disk_handoff.as_ref(),
        )
        .await;
    }
    record.bytes_out = processed.bytes.len();
    let mut response = if meta {
        meta_document(state, processed, sealed, accept).await
    } else {
        respond(
            processed.format,
            processed.quality,
            bytes::Bytes::from(processed.bytes),
            max_age,
            state.client_hints,
        )
    };
    if !meta {
        attach_cache_headers(&mut response, cache_status, &groups, sealed);
    }
    if used_fallback {
        record.fallback = true;
        response.headers_mut().insert(
            "x-crabcrop-fallback",
            axum::http::HeaderValue::from_static("1"),
        );
        // A fallback is a placeholder for a transient failure:
        // it must never be pinned downstream for the immutable
        // year (audit PL-2).
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    if debug {
        // A diagnostic is computed for this request only: no
        // downstream cache may pin it.
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    // A meta response serves a document about the render,
    // not the render: it carries no validator for the
    // image bytes (and is no-store besides).
    let render_etag = if meta { None } else { Some(render_etag) };
    (response, record, render_etag)
}

/// The stages of the miss path that can fail (issue #479):
/// [`load_source`], [`load_watermarks`], [`build_job`], and
/// [`render_with_fallback`], which takes the gates through
/// [`acquire_job_permits`]. A failure is the plain
/// request's answer, which [`serve_miss`] maps for a sealed
/// one. These stages box it: a bare `Response` is 128
/// bytes, too large for an `Err` (clippy's
/// `result_large_err`).
pub(in crate::server) async fn render_miss(
    state: &AppState,
    work: &MissWork,
    record: &mut RequestRecord,
) -> Result<Rendered, Box<Response>> {
    let accept = work.accept.as_deref();
    let directives = &work.directives;
    let wants_video = directives.explicit_format.is_some_and(|f| f.is_video());
    // A `frame:` segment makes the source a video (issue
    // #214): only then may the loader accept a `video/*`
    // Content-Type for it. A `page:` segment does the
    // same for `application/pdf` (issue #215).
    let wants_frame = directives.frame.is_some();
    let wants_page = directives.page.is_some();
    record.load_ran = true;
    // Every byte this miss buffers, the source, its
    // watermarks, a fallback, is charged to the source
    // buffer budget as it arrives (issue #403), and the
    // charge travels with the bytes into the render job,
    // which returns it when it finishes. A burst of distinct
    // misses therefore waits here, before the gates below,
    // instead of each buffering a whole source first; a miss
    // whose wait for budget runs out sheds like a full gate.
    // Nothing is charged before an origin answers, and an
    // origin that answers and then falls behind, stalling or
    // trickling, holds only what it has delivered (see
    // `SourceCharge`).
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let admission = crate::loader::MediaAdmission {
        video: wants_frame,
        pdf: wants_page,
    };
    let (bytes, used_fallback) =
        load_source(state, &work.source_url, admission, &charge, accept, record).await?;
    let watermark_sources =
        load_watermarks(state, &directives.watermark_urls, &charge, accept, record).await?;

    record.process_ran = true;
    let process_start = Instant::now();
    let needs = GateNeeds {
        ai: ai_gate_applies(
            directives,
            state.providers.face.is_some(),
            state.providers.focal.is_some(),
        ),
        video: wants_video || wants_frame || wants_page,
        // The animation decision covers animated GIF
        // sources rendered to WebP as well as animated WebP
        // sources, and counts the negotiated format, not
        // only an explicit one (issue #405); it is made
        // once, so the gate, the lane, and a fallback retry
        // agree.
        anim: wants_animation_gate(&bytes, directives, work.preferred),
    };
    // The lane this run draws its permit from (issue
    // #324): long-running work goes to the slow lane, so
    // its saturation sheds slow requests only and stills
    // keep their full capacity; everything else takes
    // the stills lane.
    let lane = if wants_slow_lane(needs.anim, wants_video, wants_frame, wants_page) {
        Lane::Slow
    } else {
        Lane::Process
    };
    let plan = RunPlan {
        lane,
        needs,
        start: process_start,
    };
    let job = build_job(state, work, watermark_sources);
    let source = JobSource {
        bytes,
        share: Some(charge),
    };
    let rendered =
        render_with_fallback(state, &job, source, used_fallback, plan, accept, record).await;
    // The source's budget share drained with its run; the
    // watermarks go now with theirs (issue #403), not after
    // the population writes and meta annotations.
    drop(job);
    rendered
}

/// Load the miss's source, charged to `charge` (issue
/// #403), timing the load into the record. Returns the
/// bytes and whether they are the configured fallback's,
/// which stands in for a source that failed to load.
async fn load_source(
    state: &AppState,
    source_url: &str,
    admission: crate::loader::MediaAdmission,
    charge: &crate::loader::SourceCharge,
    accept: Option<&str>,
    record: &mut RequestRecord,
) -> Result<(Vec<u8>, bool), Box<Response>> {
    let load_start = Instant::now();
    let loaded = match state
        .loader
        .load_media_charged(source_url, admission, charge)
        .await
    {
        Ok(b) => (b, false),
        Err(e) => {
            record.load = load_start.elapsed();
            // A policy refusal (Forbidden) is a decision, not a
            // failure: it never falls back. Neither does an
            // exhausted source budget or lookup slots (issue
            // #403): that is overload, and the fallback would
            // need budget too. Nor does source text that does
            // not parse (issue #467): a 400 stays honest, like
            // a grammar failure. Any other load failure
            // degrades to the configured fallback when there
            // is one.
            let fb = if e.is_busy()
                || matches!(e, LoadError::Forbidden(_) | LoadError::InvalidSource(_))
            {
                None
            } else {
                fallback_source_bytes(state, charge).await
            };
            match fb {
                Some(b) => (b, true),
                None => {
                    // A budget shed is overload, like a full
                    // gate, not a loader error (issue #403).
                    if !e.is_busy() {
                        record.fail = FailStage::Load;
                    }
                    return Err(Box::new(load_error_response(e, accept)));
                }
            }
        }
    };
    record.load = load_start.elapsed();
    record.bytes_in = loaded.0.len();
    Ok(loaded)
}

/// Load the miss's watermark images.
///
/// Watermarks are fetched here, before the engine runs:
/// each referenced image goes through the same loader (and
/// its bounds and source policy) as sources, and a
/// watermark that fails to load fails the request rather
/// than being silently skipped. Only the bytes travel
/// with the job: the decode happens inside the gated
/// blocking job with the rest of the engine work, so a
/// watermark that does not decode fails the request from
/// there with the same engine error (audit R2-16).
/// Watermark loads charge the same budget, against the
/// request's one wait (issue #403); their share is split
/// off to travel with the watermark bytes.
async fn load_watermarks(
    state: &AppState,
    urls: &[String],
    charge: &crate::loader::SourceCharge,
    accept: Option<&str>,
    record: &mut RequestRecord,
) -> Result<Arc<WatermarkSources>, Box<Response>> {
    let mut watermark_images: Vec<(String, Vec<u8>)> = Vec::new();
    for wm_url in urls {
        record.load_ran = true;
        let wm_bytes = match state
            .loader
            .load_media_charged(wm_url, crate::loader::MediaAdmission::default(), charge)
            .await
        {
            Ok(b) => b,
            Err(e) => {
                if !e.is_busy() {
                    record.fail = FailStage::Load;
                }
                return Err(Box::new(load_error_response(e, accept)));
            }
        };
        watermark_images.push((wm_url.to_string(), wm_bytes));
    }
    let watermark_bytes: usize = watermark_images.iter().map(|(_, b)| b.len()).sum();
    Ok(Arc::new(WatermarkSources {
        images: watermark_images,
        share: Some(charge.split_off(watermark_bytes)),
    }))
}

/// The engine job for a miss: the request's operations and
/// directives with the server's providers and settings.
pub(in crate::server) fn build_job(
    state: &AppState,
    work: &MissWork,
    watermark_sources: Arc<WatermarkSources>,
) -> ProcessJob {
    let directives = &work.directives;
    // A width hint applies only when the URL sizes nothing
    // itself: it synthesizes the width-only size op a client
    // URL would have carried, ahead of the URL's own ops.
    // The op carries `no_upscale` semantics (issue #403): a
    // hint says how wide the client displays the image, not
    // that the server should invent pixels, so a hint past
    // the source returns the source width, and the DPR hint
    // runs into the same cap. The directive is request-wide,
    // but the URL has no size op of its own here, so the
    // synthesized op is the only resize it can bound.
    let job_ops = match work.hinted_width {
        Some(w) => {
            let mut combined = width_hint_ops(w, directives.no_upscale);
            combined.extend(work.ops.iter().cloned());
            combined
        }
        None => work.ops.clone(),
    };
    ProcessJob {
        save_data: work.save_data,
        preserve_metadata: state.preserve_metadata,
        ops: job_ops,
        preferred: work.preferred,
        watermark_sources,
        placeholders: work.placeholders,
        providers: state.providers.clone(),
        ai_edited: directives.removebg
            || directives.studio
            || directives.remove_region
            || directives.upscale
            || directives.needs_face_model
            || directives.has_class_redaction(),
        want_provenance: work.meta,
        dpr: work.dpr_hint,
        ssim_target: state.ssim_target,
        brand_color: state.brand_color,
        avif_speed: state.avif_speed,
        avif_tiles: state.avif_tiles,
        jxl_speed: state.jxl_speed,
        progress: work.progress.clone(),
    }
}

/// Run the miss's job under its gates, and once more on the
/// configured fallback when the source loaded but did not
/// decode ([`retry_on_fallback`]). Times the process phase
/// into the record.
async fn render_with_fallback(
    state: &AppState,
    job: &ProcessJob,
    source: JobSource,
    mut used_fallback: bool,
    plan: RunPlan,
    accept: Option<&str>,
    record: &mut RequestRecord,
) -> Result<Rendered, Box<Response>> {
    // The request's one process deadline starts with the
    // process phase, so the gate waits and the run itself
    // share a single budget (run_process takes it as a
    // parameter).
    let deadline = plan.start + state.process_timeout;
    let permits = acquire_job_permits(state, plan.needs)
        .await
        .map_err(|busy| Box::new(busy.response(accept)))?;
    let processed = match run_process(state, job, source, plan.lane, deadline, permits).await {
        Ok(p) => p,
        // A first run never finds its client gone (see
        // `ProcessFailure::ClientGone`); were it to, nobody
        // would receive the busy answer.
        Err(ProcessFailure::Busy | ProcessFailure::ClientGone) => {
            return Err(Box::new(match plan.lane {
                Lane::Slow => busy_response(
                    "server busy: long-running processing lane is full",
                    5,
                    accept,
                ),
                Lane::Process => {
                    busy_response("server busy: too many images processing at once", 1, accept)
                }
            }));
        }
        Err(ProcessFailure::Panicked) => {
            record.fail = FailStage::Engine;
            return Err(Box::new(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "processing_failed",
                "image processing failed".to_string(),
                accept,
            )));
        }
        Err(ProcessFailure::TimedOut) => {
            record.fail = FailStage::Engine;
            return Err(Box::new(error_response(
                StatusCode::GATEWAY_TIMEOUT,
                "processing_timeout",
                "processing timed out".to_string(),
                accept,
            )));
        }
        Err(ProcessFailure::Engine(e)) => {
            record.process = plan.start.elapsed();
            // A source that loads but does not decode is the
            // other fallback case. Other engine errors are
            // properties of the request or the configuration,
            // not of the source, and stay honest.
            let retry = if !used_fallback && matches!(e, EngineError::Decode(_)) {
                retry_on_fallback(state, job, plan, deadline).await
            } else {
                None
            };
            match retry {
                Some(p) => {
                    used_fallback = true;
                    record.process = plan.start.elapsed();
                    p
                }
                None => {
                    record.fail = FailStage::Engine;
                    return Err(Box::new(engine_error_response(e, accept)));
                }
            }
        }
    };
    record.process = plan.start.elapsed();
    Ok(Rendered {
        processed,
        used_fallback,
    })
}

/// The fallback run for a source that loaded but did not
/// decode: the same job on the configured fallback's bytes,
/// under the same deadline. `None` leaves the original
/// engine error standing: no fallback, no budget for it, a
/// full gate, a failed run, or a client that has gone (see
/// [`MissProgress`]), before the run or by the time its
/// lane comes free.
async fn retry_on_fallback(
    state: &AppState,
    job: &ProcessJob,
    plan: RunPlan,
    deadline: Instant,
) -> Option<crate::engine::ProcessedImage> {
    // A fallback is never stored, so for a client that has
    // gone it would be work nobody receives: it is not
    // fetched or charged, and it waits for no gate.
    if state.fallback.is_none() || !job.progress.may_start_render() {
        return None;
    }
    // The first run's source share drained with its job, so
    // the fallback is charged afresh (issue #403), with a
    // wait of its own like each gate below; the watermarks
    // keep the share they travel with. No budget leaves the
    // original engine error standing, like a full gate.
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let bytes = fallback_source_bytes(state, &charge).await?;
    // The fallback run draws on the same gates as the first:
    // each kind of work re-acquires its permit (the first
    // run's permits drained with its job), and a full gate
    // leaves the original engine error standing.
    let permits = acquire_job_permits(state, plan.needs).await.ok()?;
    let source = JobSource {
        bytes,
        share: Some(charge),
    };
    run_process(state, job, source, plan.lane, deadline, permits)
        .await
        .ok()
}
