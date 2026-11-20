use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

use super::wants_meta;
use crate::engine::EngineError;
use crate::loader::LoadError;
use crate::url::ImageFormat;

#[cfg(doc)]
use super::admission::Refusal;
#[cfg(doc)]
use super::image_vary;

/// The `Accept` header's value, for [`error_response`].
pub(in crate::server) fn accept_of(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// Quote one string for a structured-fields string list
/// (RFC 8941): backslash and quote are escaped.
fn sf_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        if ch == '\\' || ch == '"' {
            out.push('\\');
        }
        out.push(ch);
    }
    out.push('"');
    out
}

/// Attach the cache observability headers (RFC 9211
/// Cache-Status, RFC 9875 Cache-Group) to an image response.
/// A sealed request gets no Cache-Group (issue #454): its
/// groups are preset names read from inside the token, and
/// listing them would show anyone who fetches the URL part
/// of the recipe the token hides. Group purges still reach
/// its entry, because the groups are recorded on the cache
/// entry, never read back from this header.
pub(in crate::server) fn attach_cache_headers(
    response: &mut Response,
    cache_status: &'static str,
    groups: &[String],
    sealed: bool,
) {
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("cache-status"),
        axum::http::HeaderValue::from_static(cache_status),
    );
    if !sealed && !groups.is_empty() {
        let list = groups
            .iter()
            .map(|g| sf_quote(g))
            .collect::<Vec<_>>()
            .join(", ");
        response.headers_mut().insert(
            axum::http::HeaderName::from_static("cache-group"),
            axum::http::HeaderValue::from_str(&list).unwrap(),
        );
    }
}

/// Build a successful image response: content type, cache
/// headers, and the auto-quality header when quality was chosen.
///
/// The cache lifetime is the immutable year unless the request
/// carried a `maxage:` directive (issue #159), which substitutes
/// its own value for that response only: the stored entry is
/// shared, headers are composed per response. `immutable` rides
/// along at any positive value (during the lifetime the body
/// will not change, so no revalidation is owed); at `max-age=0`
/// it is dropped, because a response that is stale on arrival
/// cannot also promise to never change, and the pair would tell
/// caches to pin what the caller just asked them to revalidate.
pub(in crate::server) fn respond(
    format: ImageFormat,
    quality: Option<u8>,
    bytes: bytes::Bytes,
    max_age: Option<u32>,
    client_hints: bool,
) -> Response {
    use axum::http::HeaderValue;
    let content_type = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Jxl => "image/jxl",
        ImageFormat::Gif => "image/gif",
        ImageFormat::Avif => "image/avif",
        ImageFormat::Mp4 => "video/mp4",
        ImageFormat::WebM => "video/webm",
    };
    let cache_control = match max_age {
        None => HeaderValue::from_static("public, max-age=31536000, immutable"),
        Some(0) => HeaderValue::from_static("public, max-age=0"),
        Some(n) => format!("public, max-age={n}, immutable").parse().unwrap(),
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(header::CACHE_CONTROL, cache_control);
    // `Vary` is set by `handle`, on every answer of the
    // image route alike. Advertise the client hints the
    // server reads, so browsers send them on later requests;
    // without the advertisement only proxies and clients
    // that set the hints manually ever benefit from them.
    // With hints off there is nothing to advertise.
    if client_hints {
        headers.insert(
            axum::http::HeaderName::from_static("accept-ch"),
            axum::http::HeaderValue::from_static("Sec-CH-Width, Sec-CH-DPR, Sec-CH-Viewport-Width"),
        );
    }
    if let Some(q) = quality {
        headers.insert(
            axum::http::HeaderName::from_static("x-crabcrop-quality"),
            q.to_string().parse().unwrap(),
        );
    }
    (headers, bytes).into_response()
}

pub(in crate::server) fn if_none_match_matches(if_none_match: &str, etag: &str) -> bool {
    let inm = if_none_match.trim();
    if inm == "*" {
        return true;
    }
    inm.split(',').any(|candidate| {
        let candidate = candidate.trim();
        candidate.strip_prefix("W/").unwrap_or(candidate) == etag
    })
}

pub(in crate::server) fn finalize_validators(
    response: &mut Response,
    etag: &str,
    if_none_match: Option<&str>,
) {
    let cacheable = response
        .headers()
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("public"));
    if !cacheable || response.status() != StatusCode::OK {
        return;
    }
    response
        .headers_mut()
        .insert(header::ETAG, etag.parse().unwrap());
    if if_none_match.is_some_and(|inm| if_none_match_matches(inm, etag)) {
        *response.status_mut() = StatusCode::NOT_MODIFIED;
        *response.body_mut() = axum::body::Body::empty();
        response.headers_mut().remove(header::CONTENT_TYPE);
        response.headers_mut().remove(header::CONTENT_LENGTH);
    }
}

/// The 503 for an exhausted source buffer budget (issue
/// #403): the same `server_busy` answer as a full gate,
/// with the stills lane's short Retry-After, since shares
/// free as soon as renders end.
fn source_budget_busy(accept: Option<&str>) -> Response {
    busy_response(SOURCE_BUDGET_BUSY, 1, accept)
}

pub(in crate::server) const SOURCE_BUDGET_BUSY: &str =
    "server busy: too many source images held in memory";

const SOURCE_LOOKUP_BUSY: &str = "server busy: too many source host lookups waiting";

/// The 503 for a load shed as overload, when `e` is one: the
/// source buffer budget's, or the host lookup slots' (see
/// [`crate::loader::LOOKUP_SLOTS`]), which carries the gates'
/// Retry-After, since a slot that stayed taken for the whole
/// wait is held by a lookup the resolver is not answering.
pub(in crate::server) fn load_busy_response(
    e: &LoadError,
    accept: Option<&str>,
) -> Option<Response> {
    match e {
        LoadError::Busy => Some(source_budget_busy(accept)),
        LoadError::LookupBusy => Some(busy_response(SOURCE_LOOKUP_BUSY, 5, accept)),
        _ => None,
    }
}

/// A 503 for a saturated gate. Retry-After tells clients
/// and CDNs when a retry has a chance, instead of letting
/// them retry straight into the same full queue; the value
/// is sized to the gate (the AI gate's jobs run seconds,
/// the process gate's run in well under one).
pub(in crate::server) fn busy_response(
    message: &'static str,
    retry_after_secs: u64,
    accept: Option<&str>,
) -> Response {
    let mut response = error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_busy",
        message.to_string(),
        accept,
    );
    response.headers_mut().insert(
        header::RETRY_AFTER,
        retry_after_secs.to_string().parse().unwrap(),
    );
    response
}

/// The uniform refusal for a sealed request (issues #255,
/// #259): once a token has opened, every failure answers
/// with the same 403, so a lifted token reveals nothing
/// about the payload it carries. The real error still
/// reaches the server log through the normal builders.
pub(in crate::server) fn sealed_refusal_response() -> Response {
    (StatusCode::FORBIDDEN, "bad signature").into_response()
}

/// `resp` for a plain request, the uniform sealed refusal
/// for a sealed one.
pub(in crate::server) fn sealed_or(sealed: bool, resp: Response) -> Response {
    if sealed {
        sealed_refusal_response()
    } else {
        resp
    }
}

/// The load failure classes: status, stable code, and the
/// generic client message, shared by the image path and
/// the explain path. The detailed cause (which can embed
/// origin-controlled strings and server-constructed store
/// URLs) goes to the server log; clients get a short
/// generic message (PL-6). A policy refusal is
/// genericized the same way the explain path does it
/// (issue #196, requirement 3): the reason can name a
/// bucket, which is the operator's fact, not the caller's.
/// A host name that resolves to a non-public address is a
/// [`LoadError::Network`] cause, so it answers exactly as a
/// name that does not resolve.
pub(in crate::server) fn load_error_parts(e: LoadError) -> (StatusCode, &'static str, String) {
    match e {
        LoadError::Forbidden(_) => (
            StatusCode::FORBIDDEN,
            "source_forbidden",
            "source refused by policy".to_string(),
        ),
        LoadError::NotFound => (
            StatusCode::NOT_FOUND,
            "source_not_found",
            "source not found".to_string(),
        ),
        LoadError::BadStatus(code) => (
            StatusCode::BAD_GATEWAY,
            "source_bad_status",
            format!("source returned status {code}"),
        ),
        LoadError::Network(_) => (
            StatusCode::BAD_GATEWAY,
            "source_fetch_failed",
            "source fetch failed".to_string(),
        ),
        // The caller's own source text is at fault, not an
        // origin (issue #467), so this is a 400 like a
        // grammar failure, not a gateway error.
        LoadError::InvalidSource(_) => (
            StatusCode::BAD_REQUEST,
            "invalid_source",
            "invalid source URL".to_string(),
        ),
        LoadError::WrongContentType(_) => (
            StatusCode::BAD_GATEWAY,
            "source_not_an_image",
            "source did not return an image".to_string(),
        ),
        LoadError::Timeout => (
            StatusCode::GATEWAY_TIMEOUT,
            "source_timeout",
            "source timed out".to_string(),
        ),
        // 502, not 413: the client's request was a valid
        // signed URL; it is the origin object that is too
        // big, like the other origin faults above.
        LoadError::TooLarge { limit } => (
            StatusCode::BAD_GATEWAY,
            "source_too_large",
            format!("source exceeds {limit} bytes"),
        ),
        // The overload answer every gate gives (issue #403).
        LoadError::Busy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            SOURCE_BUDGET_BUSY.to_string(),
        ),
        LoadError::LookupBusy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            SOURCE_LOOKUP_BUSY.to_string(),
        ),
    }
}

pub(in crate::server) fn load_error_response(e: LoadError, accept: Option<&str>) -> Response {
    if let Some(busy) = load_busy_response(&e, accept) {
        return busy;
    }
    stderr_line!("load error: {e}");
    let (status, code, message) = load_error_parts(e);
    error_response(status, code, message, accept)
}

/// One error response in the v2 contract: the plain-text
/// body every client has always gotten, or the v2 JSON
/// envelope (`{"error":{"code","message"}}`, the code
/// being the failure's stable name) when the client asked
/// for JSON, the same test the meta surface uses.
/// Grammar failures carry the parser's codes; operational
/// failures carry the codes assigned at their builders
/// (issue #356). Codes are frozen once published, like
/// the parser's: a failure class may be renamed in code,
/// but its code never changes. What this builds is never
/// sent to a sealed request for a payload failure, which
/// answers the uniform refusal instead: the image path's
/// [`Refusal::answer`] skips this builder, and the miss
/// path's [`sealed_or`] drops the answer its failure
/// built. Every error answer leaves with `Cache-Control:
/// no-store`, which the router's response-header layer
/// adds on every route, and on the image route with the
/// same `Vary` as an image ([`image_vary`]).
pub(in crate::server) fn error_response(
    status: StatusCode,
    code: &str,
    message: String,
    accept: Option<&str>,
) -> Response {
    if wants_meta(accept) {
        let body = serde_json::json!({
            "error": { "code": code, "message": message }
        });
        (status, axum::Json(body)).into_response()
    } else {
        (status, message).into_response()
    }
}

/// Cap for a parse error echoed to a client (issue
/// #268): the parser's message embeds the offending
/// segment verbatim, and a segment is bounded only by
/// the URL length cap, so a 10 KB segment produced a
/// 10 KB error body. 200 bytes keeps the useful head of
/// every message shape; the full text stays in the
/// server log through the record.
const MAX_ERROR_ECHO: usize = 200;

fn truncate_echo(message: &str) -> String {
    if message.len() <= MAX_ERROR_ECHO {
        return message.to_string();
    }
    let mut end = MAX_ERROR_ECHO;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &message[..end])
}

/// The parts of an error answer before [`error_response`]
/// shapes them for the client's `Accept`: the status, the
/// stable code, and the message. A refusal that carries
/// them leaves the answer to the one place that knows how
/// the request may be answered (see [`Refusal::answer`]).
pub(in crate::server) type ErrorParts = (StatusCode, &'static str, String);

/// The refusal for an image remainder that begins with
/// [`crabcrop_url::RESERVED_SEGMENT`]. Every derived subkey
/// (purge, purge-group, upload, seal, cache entry) is the
/// HMAC of a label under that segment, so no image URL may
/// begin with it, and no source can: it names no scheme the
/// loader fetches.
pub(in crate::server) fn reserved_remainder_parts() -> ErrorParts {
    (
        StatusCode::BAD_REQUEST,
        "reserved_segment",
        format!(
            "an image URL may not begin with the reserved segment {}",
            crabcrop_url::RESERVED_SEGMENT
        ),
    )
}

/// The refusal for a signature that does not verify, and
/// for a sealed spelling on the explain route.
pub(in crate::server) fn bad_signature_parts() -> ErrorParts {
    (
        StatusCode::FORBIDDEN,
        "bad_signature",
        "bad signature".to_string(),
    )
}

/// A grammar failure's parts: the parser's code and its
/// (truncated) message.
pub(in crate::server) fn parse_failure_parts(e: &crate::url::ParseError) -> ErrorParts {
    (
        StatusCode::BAD_REQUEST,
        e.code(),
        truncate_echo(&e.to_string()),
    )
}

/// The engine failure classes: status, stable code, and
/// the generic client message.
fn engine_error_parts(e: EngineError) -> (StatusCode, &'static str, String) {
    match e {
        EngineError::Decode(_) => (
            StatusCode::BAD_GATEWAY,
            "decode_failed",
            "source image could not be decoded".to_string(),
        ),
        EngineError::Encode(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "encode_failed",
            "image encoding failed".to_string(),
        ),
        EngineError::Timeout => (
            StatusCode::GATEWAY_TIMEOUT,
            "processing_timeout",
            "processing timed out".to_string(),
        ),
        EngineError::Unsupported(what) => (StatusCode::BAD_REQUEST, "unsupported_operation", what),
        EngineError::NotConfigured(what) => (StatusCode::NOT_IMPLEMENTED, "not_configured", what),
        EngineError::Inference(crate::ai::AiError::BadOutput(_)) => (
            StatusCode::BAD_GATEWAY,
            "inference_bad_output",
            "model produced unexpected output".to_string(),
        ),
        EngineError::Inference(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "image inference failed".to_string(),
        ),
        EngineError::Render(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "render_failed",
            "image rendering failed".to_string(),
        ),
        // The encoder is the operator's own program: its
        // failure is an upstream fault (502). The body is
        // generic, like every other class here: the cause
        // (the configured program path, the encoder's own
        // stderr) stays in the log line above, where the
        // operator reads it, and never reaches the client
        // (audit R2-30).
        EngineError::Video(_) => (
            StatusCode::BAD_GATEWAY,
            "video_encode_failed",
            "video encoding failed".to_string(),
        ),
        EngineError::Frame(_) => (
            StatusCode::BAD_GATEWAY,
            "frame_extract_failed",
            "video frame extraction failed".to_string(),
        ),
        EngineError::Pdf(_) => (
            StatusCode::BAD_GATEWAY,
            "pdf_render_failed",
            "PDF rendering failed".to_string(),
        ),
        // A dependency outage, not a broken encoder (issue
        // #467): the output was rendered, but the timestamp
        // authority signing needs did not answer, and a
        // retry may find it back.
        EngineError::ProvenanceUnavailable(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "provenance_unavailable",
            "provenance signing is unavailable".to_string(),
        ),
    }
}

/// `Retry-After` on a `provenance_unavailable` answer, in
/// seconds. Longer than a gate's: an authority outage
/// outlasts a queue, and every retry that meets it holds a
/// process permit for up to the authority's 15-second
/// bound before it fails again.
const PROVENANCE_RETRY_AFTER_SECS: u64 = 30;

pub(in crate::server) fn engine_error_response(e: EngineError, accept: Option<&str>) -> Response {
    stderr_line!("engine error: {e}");
    let retry_after = matches!(e, EngineError::ProvenanceUnavailable(_));
    let (status, code, message) = engine_error_parts(e);
    let mut response = error_response(status, code, message, accept);
    if retry_after {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            PROVENANCE_RETRY_AFTER_SECS.to_string().parse().unwrap(),
        );
    }
    response
}
