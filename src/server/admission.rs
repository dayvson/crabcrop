use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use axum::response::Response;

use super::request::quantize_width;
use super::response::{
    ErrorParts, bad_signature_parts, error_response, parse_failure_parts, reserved_remainder_parts,
    sealed_refusal_response,
};
use super::{AppState, meta_cache_headers, wants_meta, wants_placeholders};
use crate::observe::RequestRecord;
use crate::url::{ImageFormat, parse_operations_with_expiry, parse_signature};

#[cfg(doc)]
use super::explain::admit_explained;
#[cfg(doc)]
use super::miss::build_job;
#[cfg(doc)]
use super::request::{cache_key_for, serve};

/// Why the image path refused a request before its cache
/// lookup (issue #479). The stages of [`serve`] return
/// every refusal as one of these, and `serve` answers it
/// in one place, [`Refusal::answer`], so a refusal added to
/// any stage keeps the sealed spelling's uniform 403
/// (issues #255, #259) without a branch of its own.
pub(in crate::server) enum Refusal {
    /// A segment in the sealed spelling that does not
    /// open: the uniform refusal, whatever the request.
    Unopened,
    /// A failure of what the remainder asks for: its own
    /// answer for a plain request, the uniform refusal for
    /// a sealed one, so an opened token is never a
    /// validity oracle.
    Payload(ErrorParts),
    /// A request for meta with meta mode off (issues #390,
    /// #453): a payload refusal whose answer, either one,
    /// carries the meta document's caching headers.
    MetaDisabled(ErrorParts),
    /// An answer both spellings give as is: the plain
    /// spelling's own signature failures, which a sealed
    /// request never reaches, and an expired URL, which an
    /// opened token answers exactly as the plain form does
    /// (`signature_expired`).
    AsIs(ErrorParts),
}

impl Refusal {
    /// The answer to this refusal, for a request in the
    /// sealed spelling or not. A sealed payload refusal
    /// answers the uniform refusal without building its
    /// own answer first.
    pub(in crate::server) fn answer(self, sealed: bool, accept: Option<&str>) -> Response {
        let own =
            |(status, code, message): ErrorParts| error_response(status, code, message, accept);
        match self {
            Refusal::Unopened => sealed_refusal_response(),
            Refusal::Payload(_) if sealed => sealed_refusal_response(),
            Refusal::MetaDisabled(_) if sealed => meta_cache_headers(sealed_refusal_response()),
            Refusal::Payload(parts) | Refusal::AsIs(parts) => own(parts),
            Refusal::MetaDisabled(parts) => meta_cache_headers(own(parts)),
        }
    }
}

/// A remainder the request may run.
pub(in crate::server) struct Opened<'a> {
    /// The remainder as presented, or as its sealed token
    /// decrypts.
    pub(in crate::server) remainder: std::borrow::Cow<'a, str>,
    /// What admitted it.
    pub(in crate::server) credential: Credential,
}

/// What admitted a remainder (see [`authenticate`]).
pub(in crate::server) enum Credential {
    /// A sealed token that opened: the request arrived in
    /// the sealed spelling (issue #255).
    Sealed,
    /// A signature that verified, with the key ID its
    /// segment named, if any (issue #143).
    Signature(Option<String>),
    /// The `unsafe` segment, on a server that allows it.
    Unsafe,
}

/// The first stage of [`serve`]: verify the signature over
/// the remainder, or open the sealed token in its slot,
/// timing either into `record.verify`.
///
/// A sealed spelling (issue #255) is intercepted before
/// the plain parse: the token sits in the signature slot,
/// and everything about it (an oversized segment, a
/// failed open, and every payload failure after a
/// successful open) answers with the one uniform refusal,
/// so a token is never a validity oracle. On a successful
/// open the decrypted remainder takes the place of the
/// presented one and flows through the exact pipeline a
/// verified plain remainder flows through: because
/// decryption produces the pre-expansion string the HMAC
/// form carries, the two spellings share one cache entry,
/// one ETag, one purge target by construction.
pub(in crate::server) fn authenticate<'a>(
    state: &AppState,
    full: &'a str,
    record: &mut RequestRecord,
) -> Result<Opened<'a>, Refusal> {
    let nolead = full.strip_prefix('/').unwrap_or(full);
    let first_segment = nolead.split('/').next().unwrap_or("");
    if crate::url::looks_sealed(first_segment)
        && crate::url::split_sealed_segment(first_segment).is_none()
    {
        // Marker-shaped but not a token this scheme opens
        // (an unknown version): the uniform refusal, never
        // the grammar's 400, so the answer cannot announce
        // that the segment was not a token at all.
        return Err(Refusal::Unopened);
    }
    if let Some((key_id, token)) = crate::url::split_sealed_segment(first_segment) {
        if first_segment.len() > crabcrop_url::MAX_SEALED_SEGMENT {
            return Err(Refusal::Unopened);
        }
        let verify_start = Instant::now();
        let opened = state.keyring.open_sealed(key_id, token);
        record.verify = verify_start.elapsed();
        // For a sealed request the open is the
        // verification the plain spelling runs below.
        return match opened {
            Some(text) => Ok(Opened {
                remainder: std::borrow::Cow::Owned(text),
                credential: Credential::Sealed,
            }),
            None => Err(Refusal::Unopened),
        };
    }
    // Grammar failures answer 400 with the parser's own
    // message as the body: it names only what the caller
    // sent, and it is the difference between a debuggable
    // rejection and an empty one.
    let (signature, key_id, rest) =
        parse_signature(full).map_err(|e| Refusal::AsIs(parse_failure_parts(&e)))?;
    // A remainder that begins with the reserved segment is
    // refused before its signature is checked, so the answer
    // is the same whatever the signature: no response depends
    // on an HMAC over a key derivation label.
    if crabcrop_url::is_reserved_remainder(rest) {
        return Err(Refusal::AsIs(reserved_remainder_parts()));
    }
    let verify_start = Instant::now();
    let verified = state
        .keyring
        .verify(key_id.as_deref(), &signature, rest, state.allow_unsafe);
    record.verify = verify_start.elapsed();
    if !verified {
        return Err(Refusal::AsIs(bad_signature_parts()));
    }
    let credential = match signature {
        crate::url::Signature::Hmac(_) => Credential::Signature(key_id),
        crate::url::Signature::Unsafe => Credential::Unsafe,
    };
    Ok(Opened {
        remainder: std::borrow::Cow::Borrowed(rest),
        credential,
    })
}

/// What a verified remainder resolves to.
pub(in crate::server) struct Resolved {
    /// The operations the engine runs: the remainder's
    /// own, with the server-wide defaults merged in.
    pub(in crate::server) ops: Vec<crate::url::Operation>,
    /// How many of `ops` are the remainder's own: the
    /// last ones, since [`crate::url::merge_default_ops`]
    /// puts a request's operations after the defaults it
    /// keeps.
    pub(in crate::server) own_len: usize,
    /// The source to fetch, a root source expanded.
    pub(in crate::server) source_url: String,
    /// The root reference the source was written as
    /// (issue #248), when it was one.
    pub(in crate::server) root: Option<crate::url::RootRef>,
    /// The cache groups (RFC 9875).
    pub(in crate::server) groups: Vec<String>,
    /// The `maxage:` cache lifetime (issue #159).
    pub(in crate::server) max_age: Option<u32>,
    /// The `exp:` the URL carries, when it carries one.
    pub(in crate::server) expiry: Option<Expiry>,
}

impl Resolved {
    /// The remainder's own operations, without the
    /// server-wide defaults.
    pub(in crate::server) fn own_ops(&self) -> &[crate::url::Operation] {
        &self.ops[self.ops.len() - self.own_len..]
    }

    /// The source as the URL wrote it: a root reference as
    /// written, never its expansion.
    pub(in crate::server) fn written_source(&self) -> String {
        match &self.root {
            Some(root) => root.written(),
            None => self.source_url.clone(),
        }
    }
}

/// A URL's `exp:` second and the clock reading it was
/// admitted at, in unix seconds.
#[derive(Clone, Copy)]
pub(in crate::server) struct Expiry {
    pub(in crate::server) at: u64,
    pub(in crate::server) checked_at: u64,
}

/// The clock in whole unix seconds, the unit of `exp:`.
pub(in crate::server) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The second stage of [`serve`]: expand and parse the
/// remainder, recording its operation summary, and refuse
/// what it may not ask for: a reserved segment, a preset
/// or grammar failure, too many operations, an unknown
/// source root, or an expired URL, in that order.
///
/// It and [`authenticate`] are the explain route's
/// admission too ([`admit_explained`]), so a refusal added
/// to either holds on both routes.
pub(in crate::server) fn resolve_request(
    state: &AppState,
    rest: &str,
    record: &mut RequestRecord,
) -> Result<Resolved, Refusal> {
    let grammar = |e: crate::url::ParseError| Refusal::Payload(parse_failure_parts(&e));
    // Defense in depth for the purge domain separation (audit
    // P-1): an image remainder may not begin with the `purge`
    // segment, so the two URL shapes can never be confused.
    // The upload segment (issue #168) is reserved the same
    // way, for the same reason.
    if matches!(
        rest.split('/').next(),
        Some("purge" | "purge-group" | "upload")
    ) {
        return Err(Refusal::Payload((
            StatusCode::BAD_REQUEST,
            "reserved_segment",
            "an image URL may not begin with a purge or upload segment".to_string(),
        )));
    }
    // A plain remainder that begins with the key derivation
    // segment never gets this far (`authenticate` refuses it
    // before verification); an opened token's can, and its
    // answer is the uniform refusal like any payload failure.
    if crabcrop_url::is_reserved_remainder(rest) {
        return Err(Refusal::Payload(reserved_remainder_parts()));
    }
    // A remainder that names no preset, the common request,
    // is parsed in one walk (issue #640): it has no groups,
    // and it expands to itself. Every other remainder, one
    // that fails included, takes the three walks below, so
    // each error still comes from the walk that owns it.
    let plain = if state.only_presets {
        None
    } else {
        crate::url::parse_without_presets(rest)
    };
    let expanded;
    let (groups, rest, (ops, source_url, expiry, max_age)) = match plain {
        Some(parsed) => (Vec::new(), rest, parsed),
        None => {
            // The cache groups (RFC 9875) are the presets the
            // signed remainder references, read before
            // expansion rewrites it.
            let groups = crate::url::preset_names(rest);
            // Preset expansion happens after verification (the
            // signature covers the preset reference, not its
            // expansion) and before everything else: parsing,
            // the cache key, and the ops summary all see the
            // expanded remainder, so editing a preset produces
            // fresh cache entries naturally. Expansion always
            // runs, even with no presets loaded: a `preset:`
            // segment is a grammar error (`unknown_preset`) on
            // every server, never source text.
            expanded = crate::url::expand_presets(rest, &state.presets, state.only_presets)
                .map_err(grammar)?;
            let parsed = parse_operations_with_expiry(&expanded).map_err(grammar)?;
            (groups, expanded.as_str(), parsed)
        }
    };
    // The operation summary is everything before the source URL.
    record.ops = rest[..rest.len() - source_url.len()]
        .trim_end_matches('/')
        .to_string();
    // Server-wide defaults (issue #145) merge here: prepended
    // to the request's own operations, so the directive
    // analysis below, the engine, and the cache key all see
    // the merged list, and the URL wins every last-wins
    // directive (format, quality, fill) by coming later. The
    // signature never covers the defaults: they are operator
    // configuration, not request content.
    let own_len = ops.len();
    let ops = crate::url::merge_default_ops(&state.default_ops, ops);
    // The operation cap (issue #469) counts this merged list,
    // the one the engine runs, and answers here: before the
    // cache, the gates, and the source fetch, so an over-cap
    // URL (a short one repeating a preset included) costs a
    // parse and nothing more.
    crate::url::check_operation_count(&ops).map_err(grammar)?;
    // So does the cap on distinct watermark and overlay URLs
    // (issue #613): each one is a fetch and a decode.
    crate::url::check_watermark_count(&ops).map_err(grammar)?;
    // A root source expands here (issue #248): after
    // verification and the ops summary (which reads the
    // source as written), before the cache key and the
    // fetch, so the short and full spellings of one source
    // are one render, sharing a cache entry and a purge
    // target. The loader's checks evaluate the expanded
    // URL unchanged, the allowlist included.
    let (source_url, root) =
        crate::url::expand_source_root(source_url, &state.source_roots).map_err(grammar)?;
    // Expiry is enforced after signature verification (the
    // segment is inside the signed remainder, so a tampered
    // expiry already failed as a bad signature) and before the
    // cache: an expired URL never serves, cached or not. The
    // comparison is exact: the URL is valid strictly before
    // its `exp` second.
    let expiry = match expiry {
        Some(at) => {
            let checked_at = unix_seconds();
            if checked_at >= at {
                return Err(Refusal::AsIs((
                    StatusCode::FORBIDDEN,
                    "signature_expired",
                    "signature expired".to_string(),
                )));
            }
            Some(Expiry { at, checked_at })
        }
        None => None,
    };
    Ok(Resolved {
        ops,
        own_len,
        source_url,
        root,
        groups,
        max_age,
        expiry,
    })
}

/// How the request's directives and headers shape its
/// response.
pub(in crate::server) struct Negotiated {
    pub(in crate::server) directives: crate::url::RequestDirectives,
    /// The format negotiated from `Accept`, when the URL
    /// names none.
    pub(in crate::server) preferred: Option<ImageFormat>,
    /// The DPR factor the response is rendered at.
    pub(in crate::server) effective_dpr: f32,
    /// The quantized width a width hint resolves to.
    pub(in crate::server) hinted_width: Option<u32>,
    /// Whether `Save-Data` applies: the engine steps the
    /// quality of a format that carries one, and the key
    /// separates the entry.
    pub(in crate::server) save_data_applies: bool,
    pub(in crate::server) meta: bool,
    pub(in crate::server) debug: bool,
    pub(in crate::server) placeholders: bool,
}

impl Negotiated {
    /// Whether a client hint gives the answer its own cache
    /// entry: a width hint sizes it, a DPR hint (not the
    /// URL's own `dpr`) scales it, or `Save-Data` applies.
    /// This is exactly when [`cache_key_for`] gives the key
    /// a hint's suffix, so the answer is set apart from the
    /// URL's hint-free answer by the same rule its cache
    /// entry is. It can still match that answer byte for
    /// byte (a width hint at or past the source width, a DPR
    /// hint on a URL that sizes nothing, `Save-Data` on PNG
    /// output), and is set apart all the same: whether the
    /// bytes differ is only known after the render.
    pub(in crate::server) fn hint_shaped(&self) -> bool {
        self.hinted_width.is_some()
            || self.save_data_applies
            || (self.directives.dpr.is_none() && self.effective_dpr != 1.0)
    }
}

/// The third stage of [`serve`]: settle the directives,
/// the client hints, and the negotiated format, and refuse
/// a meta request this server or this output cannot
/// answer.
pub(in crate::server) fn negotiate(
    state: &AppState,
    ops: &[crate::url::Operation],
    accept: Option<&str>,
    dpr_hint: Option<f32>,
    width_hint: Option<u32>,
    save_data: bool,
) -> Result<Negotiated, Refusal> {
    // One pass over the ops list settles every directive
    // the handler reads (issue #136).
    let directives = crate::url::RequestDirectives::analyze(ops);

    // The DPR client hint settles sizing when the URL has no
    // explicit `dpr` directive; the effective factor is part of
    // what the response is, so it joins the cache key whenever
    // it changes the output. A hinted factor the output limits
    // refuse steps down in the engine (issue #529), and that
    // render is stored under the hinted factor's key: the hint
    // and the URL settle the step together, so the entry
    // answers that hint alone and never the hint-less URL's.
    let effective_dpr = directives.dpr.or(dpr_hint).unwrap_or(1.0);

    // The quantized width a width hint resolves to, when the
    // ops size nothing themselves. It is part of what the
    // response is, so it joins the cache key below. A hint
    // is advisory (issue #615): when the ops it adds would
    // take the list past the operation cap the URL passed,
    // it is dropped, and the request renders and caches as
    // the hint-less URL does instead of failing in the
    // engine after the fetch.
    let hinted_width = if directives.has_size_operation {
        None
    } else {
        width_hint
            .map(|w| quantize_width(w, state.width_step))
            .filter(|&w| {
                let added = width_hint_ops(w, directives.no_upscale);
                crate::url::operation_count(ops) + crate::url::operation_count(&added)
                    <= crate::url::MAX_OPERATIONS
            })
    };
    // Save-Data applies when the hint is present and the URL
    // names no quality directive. This one decision drives
    // the engine's step (`MissWork::save_data`), the key's
    // separate entry, and the private marking, so the three
    // cannot disagree: a per-format list that names no value
    // for the output format still counts as a directive, and
    // the engine does not step it either.
    let save_data_applies = save_data && !directives.quality_directive;

    let preferred = negotiate_format(accept);
    let meta = wants_meta(accept);
    if meta && !state.meta_enabled {
        // Meta mode is switched off on this deployment
        // (issue #390), as it is by default (issue #453):
        // refuse specifically, like the metrics endpoint
        // does, rather than serving a render the caller did
        // not ask for. The refusal answers the URL's JSON
        // representation, so it carries the meta document's
        // caching headers: a cache that stored it under the
        // bare URL would serve it to every image viewer.
        return Err(Refusal::MetaDisabled((
            StatusCode::FORBIDDEN,
            "meta_disabled",
            "meta mode is disabled on this server: set CRABCROP_META_MODE=1 to enable it"
                .to_string(),
        )));
    }
    let placeholders = meta && wants_placeholders(accept);
    let wants_video = directives.explicit_format.is_some_and(|f| f.is_video());
    // Meta describes an image; a video result is not one.
    // Refuse specifically rather than describing encoded
    // video bytes as if they were a still image. The
    // message echoes the payload's own format argument, so
    // a sealed request answers the uniform refusal instead
    // (issue #255).
    if meta && wants_video {
        return Err(Refusal::Payload((
            StatusCode::BAD_REQUEST,
            "meta_not_available",
            format!(
                "meta is not available for video output: format({}) serves encoded video, not an image",
                crate::url::format_text(directives.explicit_format.unwrap())
            ),
        )));
    }
    // The debug view (issue #164) is a diagnostic computed
    // from this request's own processing: it bypasses the
    // cache in both directions (the meta precedent) and is
    // served no-store. Meta cannot describe it either: the
    // bytes are a marked-up canvas, not the transform result
    // the meta document reports on.
    let debug = directives.debug_focal;
    if meta && debug {
        return Err(Refusal::Payload((
            StatusCode::BAD_REQUEST,
            "meta_not_available",
            "meta is not available for debug output: debug:focal serves a diagnostic image, not a transform result"
                .to_string(),
        )));
    }
    Ok(Negotiated {
        directives,
        preferred,
        effective_dpr,
        hinted_width,
        save_data_applies,
        meta,
        debug,
        placeholders,
    })
}

/// The operations a width hint puts in front of the URL's
/// own: the width-only size op, plus `no_upscale` unless
/// the request already has it. [`negotiate`] counts them
/// against the operation cap before it keeps the hint
/// (issue #615), and [`build_job`] adds them.
pub(in crate::server) fn width_hint_ops(
    width: u32,
    no_upscale: bool,
) -> Vec<crate::url::Operation> {
    let mut ops = vec![crate::url::Operation::Size(crate::url::ImageSize {
        width: crate::url::Dimension::Pixels(width),
        height: crate::url::Dimension::Proportional,
        flip_horizontal: false,
        flip_vertical: false,
    })];
    if !no_upscale {
        ops.push(crate::url::Operation::Filters(vec![
            crate::url::Filter::Plain(crate::url::PlainFilter::NoUpscale),
        ]));
    }
    ops
}

/// Pick a format from the client's `Accept` header. Among
/// the negotiable formats (AVIF, JPEG XL, WebP) the client
/// accepts, the highest q-value wins (Q8); ties break by
/// the server preference order (AVIF, then JXL, then
/// WebP). A missing q means 1.0, q=0 is a refusal, and a
/// format named twice is governed by its first entry.
/// `None` keeps the detected source format. Wildcards do
/// not count as an offer: a bare `*/*` from a generic HTTP
/// client should not trigger the expensive AVIF encode.
pub(in crate::server) fn negotiate_format(accept: Option<&str>) -> Option<ImageFormat> {
    let accept = accept?;
    let (mut avif, mut jxl, mut webp) = (None, None, None);
    for part in accept.split(',') {
        let mut segments = part.split(';');
        let media = segments.next().unwrap_or("").trim();
        let slot = match media {
            "image/avif" => &mut avif,
            "image/jxl" => &mut jxl,
            "image/webp" => &mut webp,
            _ => continue,
        };
        if slot.is_some() {
            continue;
        }
        let q = segments
            .filter_map(|p| {
                p.trim()
                    .strip_prefix("q=")
                    .and_then(|v| v.trim().parse::<f32>().ok())
            })
            .next()
            .unwrap_or(1.0);
        *slot = Some(q);
    }
    // Preference order is the array order; a strictly
    // greater q is required to displace the incumbent, so
    // ties go to the earlier (more preferred) format.
    let mut best: Option<(ImageFormat, f32)> = None;
    for (format, q) in [
        (ImageFormat::Avif, avif),
        (ImageFormat::Jxl, jxl),
        (ImageFormat::WebP, webp),
    ] {
        if let Some(q) = q
            && q > 0.0
            && best.is_none_or(|(_, best_q)| q > best_q)
        {
            best = Some((format, q));
        }
    }
    best.map(|(format, _)| format)
}
