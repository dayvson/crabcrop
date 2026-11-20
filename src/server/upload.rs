use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

use super::purge::purge_source_everywhere;
use super::response::{ErrorParts, SOURCE_BUDGET_BUSY, busy_response, error_response};
use super::{AppState, PROCESS_GATE_WAIT, SOURCE_BUDGET_WAIT};
use crate::engine::EngineError;
use crate::loader::LoadError;
use crate::observe::RequestLog;
use crate::url::parse_signature;

/// One structured line per upload request, the purge
/// precedent (audit PL-8): the write endpoint must be
/// auditable. The name is hashed for the same reason the
/// purge target is: a raw name is caller-chosen text.
/// Like a purge's, the line is on standard output before
/// the upload is answered.
async fn upload_log(log: &RequestLog, name: &str, outcome: &str) {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(name.as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    log.event(&format!(
        "{{\"event\":\"upload\",\"target\":\"{hash}\",\"outcome\":\"{outcome}\"}}"
    ))
    .await;
}

/// How long an upload body may take before its pace counts:
/// the body must arrive within this long plus one second for
/// every [`crate::loader::MIN_BODY_RATE`] bytes (64 KiB) it
/// has delivered (see [`upload_body_due`]).
pub const UPLOAD_BODY_GRACE: Duration = Duration::from_secs(10);

/// When an upload body whose read started at `started`, and
/// which has delivered `received` bytes, must have delivered
/// more: [`UPLOAD_BODY_GRACE`] plus one second per
/// [`crate::loader::MIN_BODY_RATE`] bytes, plus `waited`, the
/// time the server itself spent waiting for budget. A body
/// that keeps 64 KiB a second after the grace never meets
/// it, and one that trickles or stalls does, so a client
/// cannot hold its connection and its budget for longer than
/// the body's own size accounts for: a body at the 10 MiB
/// default cap is done within 10 + 160 seconds, plus the
/// budget wait.
fn upload_body_due(
    started: tokio::time::Instant,
    received: usize,
    waited: Duration,
) -> tokio::time::Instant {
    let paced = (received as u64).saturating_mul(1000) / crate::loader::MIN_BODY_RATE as u64;
    started + UPLOAD_BODY_GRACE + Duration::from_millis(paced) + waited
}

/// Read an upload body of at most `cap` bytes, charging it
/// to `charge` as it arrives (issue #403), in steps so a
/// trickling client does not touch the budget per frame.
/// [`LoadError::TooLarge`] past the cap,
/// [`LoadError::Network`] for an unreadable body,
/// [`LoadError::Timeout`] for a body that falls behind
/// [`upload_body_due`], and [`LoadError::Busy`] when the
/// request's wait for budget ran out.
pub(in crate::server) async fn read_upload_body(
    mut body: axum::body::Body,
    cap: usize,
    charge: &crate::loader::SourceCharge,
) -> Result<Vec<u8>, LoadError> {
    use axum::body::HttpBody as _;
    const STEP: usize = 256 * 1024;
    let started = tokio::time::Instant::now();
    let mut waited = Duration::ZERO;
    let mut out = Vec::new();
    loop {
        let due = upload_body_due(started, out.len(), waited);
        let next = std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx));
        let frame = match tokio::time::timeout_at(due, next).await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(_) => return Err(LoadError::Timeout),
        };
        let frame = frame.map_err(|e| LoadError::Network(e.to_string()))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        let received = out.len() + data.len();
        if received > cap {
            return Err(LoadError::TooLarge { limit: cap as u64 });
        }
        let step = received.div_ceil(STEP).saturating_mul(STEP);
        let waiting = tokio::time::Instant::now();
        charge.grow_to(step.min(cap).max(received)).await?;
        waited += waiting.elapsed();
        out.extend_from_slice(&data);
    }
    charge.shrink_to(out.len());
    Ok(out)
}

/// The outcome of running upload validation under the
/// process gate.
enum UploadValidation {
    /// The bytes decode as an image this server can
    /// serve; the prepared form (what to store, and the
    /// media kind it validated as) travels with the
    /// verdict (issue #192).
    Valid(crate::engine::PreparedUpload),
    /// The bytes do not decode; the engine's error, kept
    /// for the server log.
    Invalid(EngineError),
    /// The process gate stayed saturated past its wait:
    /// the upload is shed with a 503, like a transform.
    Busy,
    /// Validation panicked on its blocking thread.
    Panicked,
}

/// Validate uploaded bytes under the process gate on a
/// blocking thread (issue #187). Validation is a real
/// decode (animated formats frame by frame, HEIC through
/// oxideav), so it is budgeted exactly like transform
/// work: without the gate, a burst of signed uploads
/// would multiply decode load past the concurrency bound
/// the rest of the server runs under (audit R2-06).
async fn run_upload_validation(state: &AppState, bytes: Vec<u8>) -> UploadValidation {
    let permit = match tokio::time::timeout(
        PROCESS_GATE_WAIT,
        state.process_gate.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        _ => return UploadValidation::Busy,
    };
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        crate::engine::prepare_upload(&bytes)
    })
    .await
    {
        Ok(Ok(prepared)) => UploadValidation::Valid(prepared),
        Ok(Err(e)) => UploadValidation::Invalid(e),
        Err(_) => UploadValidation::Panicked,
    }
}

/// `PUT /upload/<signature>/<name>`: store one original
/// (issue #168). The route is always registered; without
/// a destination every request answers
/// `404 uploads_not_configured` before anything else is
/// checked (issue #467).
///
/// The signature is an HMAC over the name under an upload
/// subkey derived from the secret (see
/// [`crate::url::verify_upload`]): only the secret holder
/// can upload, an image or purge signature can never be
/// replayed here, and `unsafe` is never honored. The
/// expiring form (issue #192) signs
/// `exp:<unix-seconds>/<name>` instead and presents as
/// `/upload/<signature>/exp:<unix-seconds>/<name>`; the
/// expiry is inside the signed string, so it can neither
/// be stripped nor extended, and an expired token is a
/// 403 like an expired image URL. The checks run cheapest
/// first: signature, expiry, name shape, body size (read
/// with the cap as a hard bound, so an oversize body is
/// never fully buffered), the overwrite policy (so a
/// refused duplicate never pays a decode), and image
/// decode through the pipeline's own validation, run
/// under the process gate like any transform. What is
/// stored is the prepared form: the sanitized document
/// for an SVG, the caller's bytes for anything else, and
/// only when the name's extension matches the media kind
/// the body validated as. With overwriting off, the
/// final write is a conditional create, so the 409 the
/// policy promises holds even against a racing upload.
/// On success the body of the response is the source
/// reference to embed in transform URLs plus a newline,
/// and the `X-Crabcrop-Source` header is the exact
/// reference: `201` for a new object, `200` when an
/// allowed overwrite replaced one.
pub(in crate::server) async fn upload_handle(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Response {
    // An upload leaves a log line: like an image request,
    // it waits while the request log's queue is full, before
    // its body is read.
    state.request_log.room().await;
    let accept = req
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let (parts, body) = req.into_parts();
    // A buffered upload is a source like a fetched one
    // (issue #403): its body is charged to the source buffer
    // budget as it arrives and the charge is held until the
    // handler is done with the bytes, so a burst of uploads
    // waits at the budget instead of buffering past it, and
    // a client that stalls mid-body holds only what it sent.
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let outcome = match receive_upload(&state, parts.uri.path(), body, &charge).await {
        Ok(outcome) | Err(outcome) => outcome,
    };
    finish_upload(&state, outcome, accept.as_deref()).await
}

/// How an upload request ended (issue #479). The stages of
/// [`receive_upload`] return every refusal as one of these,
/// and [`finish_upload`] answers it, so the count on
/// `/metrics`, the audit line (audit PL-8), and the answer
/// are made in one place.
enum UploadOutcome {
    /// Refused before the signature check (no destination,
    /// a bad path, a sealed spelling, an unparsable
    /// signature), or for a verified remainder that does
    /// not split into an expiry and a name: answered, and
    /// neither counted nor logged.
    Unaudited(ErrorParts),
    /// Refused from the signature check on: counted, and
    /// logged under `target` (the remainder as presented
    /// for a bad signature, the name after it) with its
    /// `rejected:` outcome.
    Rejected {
        target: String,
        outcome: &'static str,
        answer: ErrorParts,
    },
    /// Shed because the source buffer budget or the process
    /// gate stayed full past its wait: counted and logged as
    /// `rejected:busy`, and answered with a 503 carrying
    /// `message`.
    Busy {
        target: String,
        message: &'static str,
    },
    /// Stored under `name`, at the source reference
    /// `source`; `replaced` when an allowed overwrite
    /// replaced an object.
    Stored {
        name: String,
        source: String,
        replaced: bool,
    },
}

impl UploadOutcome {
    fn rejected(target: &str, outcome: &'static str, answer: ErrorParts) -> Self {
        UploadOutcome::Rejected {
            target: target.to_string(),
            outcome,
            answer,
        }
    }
}

/// Count and log an upload's outcome (audit PL-8), then
/// answer it. A stored upload that replaced an object
/// purges the old one's derivatives before it answers.
async fn finish_upload(
    state: &Arc<AppState>,
    outcome: UploadOutcome,
    accept: Option<&str>,
) -> Response {
    match outcome {
        UploadOutcome::Unaudited((status, code, message)) => {
            error_response(status, code, message, accept)
        }
        UploadOutcome::Rejected {
            target,
            outcome,
            answer: (status, code, message),
        } => {
            state.metrics.record_upload();
            upload_log(&state.request_log, &target, outcome).await;
            error_response(status, code, message, accept)
        }
        UploadOutcome::Busy { target, message } => {
            state.metrics.record_upload();
            upload_log(&state.request_log, &target, "rejected:busy").await;
            busy_response(message, 1, accept)
        }
        UploadOutcome::Stored {
            name,
            source,
            replaced,
        } => {
            state.metrics.record_upload();
            upload_log(
                &state.request_log,
                &name,
                if replaced { "replaced" } else { "stored" },
            )
            .await;
            if replaced {
                // An overwrite replaces the source every
                // cached derivative was rendered from: purge
                // them all, in every tier, so the next
                // request re-renders from the new original
                // instead of serving the stale transform
                // (audit R2-07). A first store has no
                // derivatives to invalidate.
                // Logged inside the purge itself, so the
                // line lands even if this request is dropped
                // while the purge runs (issue #347).
                let _removed = purge_source_everywhere(state, &source).await;
            }
            let status = if replaced {
                StatusCode::OK
            } else {
                StatusCode::CREATED
            };
            let mut response = (status, format!("{source}\n")).into_response();
            response.headers_mut().insert(
                axum::http::HeaderName::from_static("x-crabcrop-source"),
                axum::http::HeaderValue::from_str(&source).unwrap(),
            );
            response
        }
    }
}

/// The stages of an upload (issue #479), cheapest first:
/// [`authorize_upload`], [`read_upload`], the overwrite
/// policy (so a refused duplicate never pays a decode),
/// [`validate_upload`], and [`store_upload`]. `Ok` is
/// always [`UploadOutcome::Stored`]; a refusal ends the
/// chain as its [`UploadOutcome`] in `Err`.
async fn receive_upload(
    state: &AppState,
    path: &str,
    body: axum::body::Body,
    charge: &crate::loader::SourceCharge,
) -> Result<UploadOutcome, UploadOutcome> {
    let (upload, name) = authorize_upload(state, path)?;
    let name = name.as_str();
    let bytes = read_upload(body, upload.max_bytes, charge, name).await?;
    // The existence check runs before validation, so an
    // upload that will be refused as a duplicate never
    // pays a full decode first (audit R2-06). The value
    // is reused after the store to pick 200 vs 201 and
    // the log outcome.
    let existed = upload.exists(name).await;
    if existed && !upload.allow_overwrite {
        return Err(upload_exists(name));
    }
    let prepared = validate_upload(state, bytes, name).await?;
    store_upload(upload, name, prepared, existed).await
}

/// The upload's first stage: a configured destination, a
/// well-formed path, a verified signature, an unexpired
/// token, and a well-formed name. Returns the destination
/// and the name.
fn authorize_upload<'s>(
    state: &'s AppState,
    path: &str,
) -> Result<(&'s crate::upload::UploadStore, String), UploadOutcome> {
    let Some(upload) = &state.upload else {
        return Err(UploadOutcome::Unaudited((
            StatusCode::NOT_FOUND,
            "uploads_not_configured",
            "uploads are not configured".to_string(),
        )));
    };
    let Some(tail) = path.strip_prefix("/upload/") else {
        return Err(UploadOutcome::Unaudited((
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /upload/<signature>/<name>".to_string(),
        )));
    };
    let full = format!("/{tail}");
    // A sealed spelling is refused wholesale (issue #255):
    // the server never unseals as a service, because an
    // endpoint that opens tokens would be an unsealing
    // oracle for anyone who lifts one off a public page.
    // Operators unseal locally (the CLI's `unseal` verb,
    // under their own secret) and then explain the plain
    // form. The refusal is the uniform one and no source
    // fetch happens for it.
    let first_segment = tail.split('/').next().unwrap_or("");
    if crate::url::looks_sealed(first_segment) {
        return Err(UploadOutcome::Unaudited((
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
        )));
    }
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return Err(UploadOutcome::Unaudited((
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
            )));
        }
    };
    // Verification runs over the remainder exactly as
    // presented: for the expiring form the `exp:` segment
    // is part of the signed string, so a tampered or
    // stripped expiry already fails here as a bad
    // signature.
    if !state
        .keyring
        .verify_upload(key_id.as_deref(), &signature, rest)
    {
        return Err(UploadOutcome::rejected(
            rest,
            "rejected:bad-signature",
            (
                StatusCode::FORBIDDEN,
                "bad_signature",
                "bad signature".to_string(),
            ),
        ));
    }
    let (expiry, name) = match crate::url::split_upload_remainder(rest) {
        Ok(v) => v,
        Err(e) => {
            return Err(UploadOutcome::Unaudited((
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
            )));
        }
    };
    // Expiry is enforced after verification and before
    // anything else, like an image URL: the comparison is
    // exact, the token valid strictly before its second.
    if let Some(exp) = expiry {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now >= exp {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:expired",
                (
                    StatusCode::FORBIDDEN,
                    "signature_expired",
                    "signature expired".to_string(),
                ),
            ));
        }
    }
    if let Err(e) = crate::upload::validate_upload_name(name) {
        return Err(UploadOutcome::rejected(
            name,
            "rejected:bad-name",
            (StatusCode::BAD_REQUEST, "invalid_upload_name", e),
        ));
    }
    Ok((upload, name.to_string()))
}

/// Read the upload's body, charged to `charge` as it
/// arrives (issue #403). An empty body is refused here: an
/// image is required.
async fn read_upload(
    body: axum::body::Body,
    max_bytes: u64,
    charge: &crate::loader::SourceCharge,
    name: &str,
) -> Result<Vec<u8>, UploadOutcome> {
    // The read is bounded by the cap itself: it fails the
    // moment the body passes it, which is the oversize
    // answer (413) without ever holding more than the cap in
    // memory. A body that cannot be read to its end (the
    // client aborted it, or framed it badly) is the
    // client's fault but not a size one: a 400 (issue #467).
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let bytes = match read_upload_body(body, cap, charge).await {
        Ok(bytes) => bytes,
        Err(LoadError::Busy) => {
            return Err(UploadOutcome::Busy {
                target: name.to_string(),
                message: SOURCE_BUDGET_BUSY,
            });
        }
        Err(LoadError::TooLarge { .. }) => {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:oversize",
                (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "upload_too_large",
                    format!("upload exceeds the {max_bytes}-byte limit"),
                ),
            ));
        }
        Err(LoadError::Timeout) => {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:too-slow",
                (
                    StatusCode::REQUEST_TIMEOUT,
                    "upload_too_slow",
                    format!(
                        "upload body arrived too slowly: it may take {} seconds plus one second \
                         for every {} KiB it sends",
                        UPLOAD_BODY_GRACE.as_secs(),
                        crate::loader::MIN_BODY_RATE / 1024
                    ),
                ),
            ));
        }
        Err(e) => {
            stderr_line!("upload body read failed: {e}");
            return Err(UploadOutcome::rejected(
                name,
                "rejected:unreadable",
                (
                    StatusCode::BAD_REQUEST,
                    "upload_unreadable",
                    "upload body could not be read".to_string(),
                ),
            ));
        }
    };
    if bytes.is_empty() {
        return Err(UploadOutcome::rejected(
            name,
            "rejected:not-an-image",
            (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "upload_not_an_image",
                "upload body is empty: an image is required".to_string(),
            ),
        ));
    }
    Ok(bytes)
}

/// The 409 for a name already taken while overwriting is
/// off: from the cheap existence check, or from the
/// conditional create that settles a race (audit R2-21).
fn upload_exists(name: &str) -> UploadOutcome {
    UploadOutcome::rejected(
        name,
        "rejected:exists",
        (
            StatusCode::CONFLICT,
            "upload_already_exists",
            "an object with this name already exists".to_string(),
        ),
    )
}

/// The bytes must decode as an image this server can
/// serve, through the pipeline's own decode branches:
/// anything else is refused before it is stored, so the
/// destination never holds an unservable object. The
/// decode runs under the process gate, like every
/// transform (see `run_upload_validation`). Then the name
/// must present the body as what it is. Returns the
/// prepared form to store.
async fn validate_upload(
    state: &AppState,
    bytes: Vec<u8>,
    name: &str,
) -> Result<crate::engine::PreparedUpload, UploadOutcome> {
    // The body moves into validation rather than being
    // copied for it, so the charge covers what is held.
    let prepared = match run_upload_validation(state, bytes).await {
        UploadValidation::Valid(prepared) => prepared,
        UploadValidation::Invalid(e) => {
            stderr_line!("upload rejected: {e}");
            return Err(UploadOutcome::rejected(
                name,
                "rejected:not-an-image",
                (
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "upload_not_an_image",
                    "upload is not an image this server can decode".to_string(),
                ),
            ));
        }
        UploadValidation::Busy => {
            return Err(UploadOutcome::Busy {
                target: name.to_string(),
                message: "server busy: too many images processing at once",
            });
        }
        UploadValidation::Panicked => {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:validation-failed",
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "validation_failed",
                    "image validation failed".to_string(),
                ),
            ));
        }
    };
    // The name must present the body as what it is: a
    // validated image may not be stored under an
    // extension belonging to another kind (or to active
    // content), because the destination may be served
    // directly (audit R2-22).
    if let Err(e) = crate::upload::check_upload_media(name, prepared.media) {
        return Err(UploadOutcome::rejected(
            name,
            "rejected:media-mismatch",
            (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "upload_media_mismatch",
                e,
            ),
        ));
    }
    Ok(prepared)
}

/// Write the prepared upload to the destination: the
/// stored outcome, or a refusal (a name claimed by a
/// racing upload, a failed write).
async fn store_upload(
    upload: &crate::upload::UploadStore,
    name: &str,
    prepared: crate::engine::PreparedUpload,
    existed: bool,
) -> Result<UploadOutcome, UploadOutcome> {
    // What lands is the prepared form, and with
    // overwriting off the write is a conditional create:
    // the exists check above is the cheap refusal, this
    // is the atomic one, so a name claimed between the
    // two still answers 409 and keeps the winner's bytes
    // (audit R2-21).
    let stored = if upload.allow_overwrite {
        upload.put(name, &prepared.stored).await
    } else {
        upload.put_new(name, &prepared.stored).await
    };
    match stored {
        Ok(()) => Ok(UploadOutcome::Stored {
            name: name.to_string(),
            source: upload.source_ref(name),
            replaced: existed,
        }),
        Err(crate::upload::UploadError::AlreadyExists) => Err(upload_exists(name)),
        Err(crate::upload::UploadError::Backend(cause)) => {
            stderr_line!("upload failed: {cause}");
            Err(UploadOutcome::rejected(
                name,
                "rejected:store-failed",
                (
                    StatusCode::BAD_GATEWAY,
                    "upload_store_unavailable",
                    "the upload store is unavailable".to_string(),
                ),
            ))
        }
        Err(crate::upload::UploadError::Io(cause)) => {
            stderr_line!("upload failed: {cause}");
            Err(UploadOutcome::rejected(
                name,
                "rejected:store-failed",
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "upload_store_failed",
                    "the upload could not be stored".to_string(),
                ),
            ))
        }
    }
}
