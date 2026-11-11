//! C2PA provenance (issue #173): verify the Content
//! Credentials a source carries, and sign outputs when the
//! operator configured a signing identity.
//!
//! The design:
//!
//! - A manifest never survives this pipeline byte-wise:
//!   decode and re-encode drop the JPEG APP11 / PNG chunk a
//!   manifest lives in, in both metadata modes (proven
//!   empirically). A signed output therefore names its
//!   source as a parent ingredient and records how the
//!   source's own credentials validated, without carrying
//!   the source's manifest store (see
//!   [`ProvenanceSigner::sign`] for why), which is the
//!   honest shape anyway: a transform invalidates the
//!   original claim's hash assertions.
//! - A valid credential proves origin, not untouched
//!   pixels. Nothing in this module or the docs may claim
//!   more.
//! - The signing identity is operator-supplied (a PEM
//!   certificate chain and key, like every other operator
//!   artifact). A chain that does not reach the C2PA trust
//!   list signs fine and validates, but public verifiers show
//!   the output as valid-not-trusted; that is the operator's
//!   choice to make, stated plainly in the README.
//! - Verification here checks signatures, not trust lists
//!   (issue #564): the reader loads no trust anchors, so a
//!   manifest reads `valid` or `invalid`, never `trusted`,
//!   whoever issued its certificate. Trust is for outside
//!   verifiers to judge.
//!
//! Every contact with the `c2pa` crate lives in this
//! module: the crate is pre-1.0 with breaking release
//! trains roughly every two months, so containing the
//! surface here keeps each migration a one-file job. The
//! pin is c2pa 0.91.7: 0.91.0 is the first release that
//! enforces the spec's exclusion rule (the October 2026
//! forgery class), and the code builds on the explicit
//! Context API from the start because the convenience
//! constructors are removed in 0.92.

use std::cell::Cell;
use std::io::Cursor;
use std::time::Duration;

use c2pa::{Builder, Context, Reader, Signer, SigningAlg, ValidationState};

/// The IPTC digital source type recorded on the edit
/// action of an AI-edited output: the pixels were changed
/// with a trained-algorithm component, not synthesized
/// wholesale.
const AI_EDITED_SOURCE_TYPE: &str =
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia";

/// The default timestamp authority bounds (issues #365,
/// #397): the SDK's own ureq resolver leaves every
/// timeout unset, so a hung authority would hang the boot
/// probe and every signed request. 5 seconds to connect,
/// 15 seconds for the whole round trip (the request and
/// reply are a few hundred bytes each).
const TSA_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const TSA_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// A timestamp authority and the bounds on one call to
/// it. The bounds are fields rather than constants so a
/// test can drive the real signing path against a stalled
/// authority in well under a second.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimestampAuthority {
    pub url: String,
    /// Time allowed to open the connection.
    pub connect_timeout: Duration,
    /// Time allowed for the whole call, from the lookup
    /// to the last byte of the reply.
    pub request_timeout: Duration,
}

impl TimestampAuthority {
    /// The authority at `url`, with the default bounds.
    pub fn new(url: String) -> Self {
        Self {
            url,
            connect_timeout: TSA_CONNECT_TIMEOUT,
            request_timeout: TSA_REQUEST_TIMEOUT,
        }
    }
}

/// A ureq agent carrying the TSA timeouts. It follows
/// no redirect: a timestamp reply is a direct answer,
/// and a single hop keeps the total timeout a bound on
/// the whole call. (The SDK's default stack follows up
/// to 10 redirects, and wrapped around this agent each
/// hop would restart the clock.)
fn tsa_agent(connect: Duration, total: Duration) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(connect))
        .timeout_global(Some(total))
        .max_redirects(0)
        .build();
    ureq::Agent::new_with_config(config)
}

/// The SDK signer and the timed agent its timestamp
/// requests go through (issue #397).
/// `create_signer::from_keys` returns a signer that leaves
/// `send_timestamp_request` to the trait default, and that
/// default builds a fresh `Context::new()`, whose resolver
/// has no timeout at all, whatever context the builder
/// signs under. So the builder's context never reached the
/// one call that talks to the authority. Each signing pass
/// signs through a [`SignPass`], which overrides that call
/// and delegates everything else unchanged.
struct TimedSigner {
    inner: c2pa::BoxedSigner,
    agent: ureq::Agent,
}

impl TimedSigner {
    /// A context whose resolver is the timed agent.
    fn context(&self) -> Context {
        Context::new().with_resolver(self.agent.clone())
    }

    /// A fresh signing pass over this signer.
    fn pass(&self) -> SignPass<'_> {
        SignPass {
            signer: self,
            authority_failed: Cell::new(false),
        }
    }
}

/// One signing pass: the [`TimedSigner`] as the SDK sees
/// it, plus a record of whether the timestamp authority
/// failed during the pass (issue #467). The record lives
/// here rather than on the shared signer, so concurrent
/// passes never see each other's failures, and it lets
/// [`ProvenanceSigner::sign`] tell an authority outage
/// apart from every other signing failure without reading
/// the SDK's error text.
struct SignPass<'a> {
    signer: &'a TimedSigner,
    authority_failed: Cell<bool>,
}

impl Signer for SignPass<'_> {
    fn sign(&self, data: &[u8]) -> c2pa::Result<Vec<u8>> {
        self.signer.inner.sign(data)
    }

    fn alg(&self) -> SigningAlg {
        self.signer.inner.alg()
    }

    fn certs(&self) -> c2pa::Result<Vec<Vec<u8>>> {
        self.signer.inner.certs()
    }

    fn reserve_size(&self) -> usize {
        self.signer.inner.reserve_size()
    }

    fn time_authority_url(&self) -> Option<String> {
        self.signer.inner.time_authority_url()
    }

    fn timestamp_request_headers(&self) -> Option<Vec<(String, String)>> {
        self.signer.inner.timestamp_request_headers()
    }

    fn timestamp_request_body(&self, message: &[u8]) -> c2pa::Result<Vec<u8>> {
        self.signer.inner.timestamp_request_body(message)
    }

    fn send_timestamp_request(&self, message: &[u8]) -> Option<c2pa::Result<Vec<u8>>> {
        let url = self.time_authority_url()?;
        // The SDK default skips the timestamp silently
        // when the request body cannot be built; an
        // operator who configured an authority gets the
        // error instead of an untimestamped signature.
        // Building the body is local work, so its failure
        // is not the authority's.
        let body = match self.timestamp_request_body(message) {
            Ok(body) => body,
            Err(e) => return Some(Err(e)),
        };
        let reply = c2pa::crypto::time_stamp::default_rfc3161_request(
            &url,
            self.timestamp_request_headers(),
            &body,
            message,
            &self.signer.context(),
        );
        if reply.is_err() {
            self.authority_failed.set(true);
        }
        Some(reply.map_err(Into::into))
    }

    fn ocsp_val(&self) -> Option<Vec<u8>> {
        self.signer.inner.ocsp_val()
    }

    fn direct_cose_handling(&self) -> bool {
        self.signer.inner.direct_cose_handling()
    }

    fn dynamic_assertions(&self) -> Vec<Box<dyn c2pa::dynamic_assertion::DynamicAssertion>> {
        self.signer.inner.dynamic_assertions()
    }
}

/// Why signing one output failed (issue #467).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignError {
    /// The timestamp authority failed this pass: it did
    /// not answer within its bounds, refused the
    /// connection, or answered with something that is not
    /// a timestamp. A dependency outage, so a retry may
    /// clear it. Carries the cause.
    Authority(String),
    /// Any other failure: the manifest, the ingredient,
    /// or the output bytes. Carries the cause.
    Failed(String),
}

/// The cause alone, either way: callers say which kind of
/// failure it was where it matters to them.
impl std::fmt::Display for SignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignError::Authority(cause) | SignError::Failed(cause) => f.write_str(cause),
        }
    }
}

impl std::error::Error for SignError {}

/// A 1x1 PNG, the boot self-test's signing target.
const SELF_TEST_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0,
    0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 0, 0, 3, 1,
    1, 0, 201, 254, 146, 239, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// The summary of one asset's manifest store, as meta
/// responses report it.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestSummary {
    /// Validation state: `valid` or `invalid`. Signatures
    /// and hashes are checked, trust lists are not, so
    /// `trusted` is never reported (see [`read_manifest`]).
    pub state: &'static str,
    /// The active manifest's signer, as a validator would
    /// show it: the certificate's common name, or its
    /// issuer when the name is absent. It is a name the
    /// certificate claims, so whoever made the asset chose
    /// it, and `valid` does not vouch for it. It is reported
    /// as one line of at most 64 characters (issue #595).
    pub signer: Option<String>,
    /// Manifests in the store (an asset edited elsewhere
    /// can carry its ingredients' stores too; an output
    /// this server signs carries its own manifest only).
    pub manifests: usize,
    /// Ingredients declared by the active manifest.
    pub ingredients: usize,
}

/// Provenance facts for one processed response, computed
/// inside the process job when the caller asked for them
/// (meta responses): what the source carried and what the
/// output carries.
#[derive(Debug, Clone, PartialEq)]
pub struct ProvenanceFacts {
    pub source: Option<ManifestSummary>,
    pub output: Option<ManifestSummary>,
    /// Whether this server signs outputs at all.
    pub signing_configured: bool,
}

/// The HTTP resolver every store read runs with: it answers
/// each request with an error and opens no connection.
/// Reading a store can otherwise reach the network on the
/// store's say: a CAWG identity assertion names its issuer
/// as a `did:web` host whose document the SDK fetches, and
/// the SDK's default resolver sets no timeout. A read runs
/// inside a render's job, which holds a render permit until
/// it returns, so what bounds the read must be local work
/// only.
struct NoNetwork;

impl c2pa::http::SyncHttpResolver for NoNetwork {
    fn http_resolve(
        &self,
        request: c2pa::http::http::Request<Vec<u8>>,
    ) -> Result<c2pa::http::http::Response<Box<dyn std::io::Read>>, c2pa::http::HttpResolverError>
    {
        Err(c2pa::http::HttpResolverError::Io(std::io::Error::other(
            format!(
                "no network request is made while reading a manifest store ({})",
                request.uri()
            ),
        )))
    }
}

/// The context every read of a manifest store runs under:
/// no network ([`NoNetwork`]) and no decoding of CAWG
/// identity assertions. Decoding one evaluates the
/// identity's own credential against its issuer, which is
/// where the network fetches come from; with no network an
/// issuer that publishes its key on the web could only read
/// as unavailable, a failure of the reader's making that
/// the SDK counts against the manifest. Undecoded, an
/// identity assertion is still an assertion the claim
/// hashes, so its bytes are checked like any other, and a
/// manifest's state is its own signature and hashes, which
/// is what meta reports and an output records about its
/// parent. Like [`Context::new`], it loads no trust
/// anchors.
fn reading_context() -> Context {
    let mut context = Context::new().with_resolver(NoNetwork);
    context.settings_mut().core.decode_identity_assertions = false;
    context
}

/// Read and validate the manifest store embedded in an
/// asset. `None` when the asset carries no manifest (or no
/// parseable one); a present-but-failing manifest reads
/// back with state `invalid`, never as absent. `format` is
/// a MIME type hint: the reader detects the container from
/// the bytes first and uses the hint only as a fallback.
///
/// The reader checks signatures and hashes, not trust
/// lists: its context loads no trust anchors (the SDK's
/// defaults carry none outside its own tests), so an
/// intact manifest reads `valid` whoever issued its
/// certificate. It makes no network request
/// ([`reading_context`]).
pub fn read_manifest(bytes: &[u8], format: &str) -> Option<ManifestSummary> {
    let mut cursor = Cursor::new(bytes);
    let reader = Reader::from_context(reading_context())
        .with_stream(format, &mut cursor)
        .ok()?;
    Some(summarize(&reader))
}

/// The summary of the store `reader` read.
fn summarize(reader: &Reader) -> ManifestSummary {
    // With no trust anchors the SDK never reaches Trusted
    // (an untrusted signer is tolerated and reads Valid),
    // and a reader that did reach it would still be making
    // a trust claim meta does not make (issue #564): meta
    // reports whether a manifest validates, and nothing
    // about the trust list.
    let state = match reader.validation_state() {
        ValidationState::Valid | ValidationState::Trusted => "valid",
        ValidationState::Invalid => "invalid",
    };
    let active = reader.active_manifest();
    let signer = active
        .and_then(|m| m.signature_info())
        .and_then(|info| info.common_name.as_deref().or(info.issuer.as_deref()))
        .and_then(display_signer);
    ManifestSummary {
        state,
        signer,
        manifests: reader.manifests().len(),
        ingredients: active.map(|m| m.ingredients().len()).unwrap_or(0),
    }
}

/// The longest signer name a summary reports, in
/// characters: the upper bound RFC 5280 sets for a common
/// name (issue #595).
const MAX_SIGNER_CHARS: usize = 64;

/// A certificate's name as a summary reports it (issue
/// #595): one line of at most [`MAX_SIGNER_CHARS`]
/// characters. The name is whatever the certificate gives,
/// so whoever made the asset chose it, and the certificate
/// parser bounds neither its length nor its characters: it
/// takes a name of any length, newlines included. Meta
/// passes the name on, and an agent may read it, so it
/// stays one short line. Control
/// characters (C0, DEL and C1), line and paragraph
/// separators, and bidirectional controls are dropped. A
/// longer name keeps its first 63 characters and ends in
/// `…`. `None` when nothing is left.
fn display_signer(name: &str) -> Option<String> {
    let mut kept = name.chars().filter(|&c| {
        !(c.is_control()
            || matches!(
                c,
                '\u{2028}'
                    | '\u{2029}'
                    | '\u{061C}'
                    | '\u{200E}'
                    | '\u{200F}'
                    | '\u{202A}'..='\u{202E}'
                    | '\u{2066}'..='\u{2069}'
            ))
    });
    let mut shown: String = kept.by_ref().take(MAX_SIGNER_CHARS).collect();
    if kept.next().is_some() {
        shown.pop();
        shown.push('…');
    }
    (!shown.is_empty()).then_some(shown)
}

/// The largest manifest store signing reads from a source
/// (issue #610): 1 MiB, room for a chain of several edits,
/// each with a full-size claim thumbnail. Reading a store
/// means parsing and validating it, which holds several
/// times its size in memory, so a larger store is not
/// parsed or validated: its ingredient names the source's
/// format and records no validation status, like a source
/// with no store.
const MAX_SOURCE_STORE_BYTES: usize = 1024 * 1024;

/// How many distinct status codes an output records per
/// category (success, informational, failure) for its
/// parent, and how long one code may be. The SDK's codes
/// are short dotted names (`claimSignature.validated`,
/// `assertion.dataHash.mismatch`), a few dozen in all; the
/// bounds keep what a parent adds to an output's manifest
/// small whatever its store holds.
const MAX_STATUS_CODES: usize = 32;
const MAX_STATUS_CODE_LEN: usize = 64;

/// The validation status an output records for its parent:
/// the status codes the SDK's validation of the source's
/// manifest store produced, as [`status_codes`] reduces
/// them. `None` when the source carries no embedded store,
/// when its store is over [`MAX_SOURCE_STORE_BYTES`], or
/// when the store does not parse; the ingredient then names
/// the source's format only. The store's bytes are copied
/// out of the container first, without parsing them, so an
/// oversized store is never parsed or validated, and the
/// reader runs only on a source that embeds a store. What
/// it validates is exactly the store that was measured,
/// against the source read as `format`: the reader is
/// handed those bytes rather than finding the store again,
/// since its own container detection can pick a different
/// handler than the hint (a text SVG whose first bytes also
/// look like another container). The read makes no network
/// request ([`reading_context`]).
fn source_status(source: &[u8], format: &str) -> Option<serde_json::Value> {
    let store = c2pa::jumbf_io::load_jumbf_from_stream(format, &mut Cursor::new(source)).ok()?;
    if store.len() > MAX_SOURCE_STORE_BYTES {
        return None;
    }
    let reader = Reader::from_context(reading_context())
        .with_manifest_data_and_stream(&store, format, Cursor::new(source))
        .ok()?;
    let results = serde_json::to_value(reader.validation_results()?).ok()?;
    Some(status_codes(&results))
}

/// Reduce the SDK's validation results for a source's store
/// to what an output records about its parent: the
/// distinct codes of each category, gathered over the
/// active manifest and every ingredient delta, so a failure
/// anywhere in the source's chain still reads as a failure.
/// URLs and explanations are left out: they name the
/// source's manifest and its assertions, which the output
/// does not carry. A code that is not a plain dotted name
/// within [`MAX_STATUS_CODE_LEN`] is dropped, and each
/// category keeps at most [`MAX_STATUS_CODES`].
fn status_codes(results: &serde_json::Value) -> serde_json::Value {
    let groups: Vec<&serde_json::Value> = std::iter::once(&results["activeManifest"])
        .chain(
            results["ingredientDeltas"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|delta| &delta["validationDeltas"]),
        )
        .collect();
    let is_code = |code: &&str| {
        !code.is_empty()
            && code.len() <= MAX_STATUS_CODE_LEN
            && code
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    let mut reduced = serde_json::Map::new();
    for category in ["success", "informational", "failure"] {
        let mut codes: Vec<&str> = Vec::new();
        let statuses = groups
            .iter()
            .flat_map(|group| group[category].as_array().into_iter().flatten());
        for code in statuses.filter_map(|status| status["code"].as_str().filter(is_code)) {
            if codes.len() == MAX_STATUS_CODES {
                break;
            }
            if !codes.contains(&code) {
                codes.push(code);
            }
        }
        let codes = codes
            .into_iter()
            .map(|code| serde_json::json!({ "code": code }))
            .collect();
        reduced.insert(category.to_string(), serde_json::Value::Array(codes));
    }
    serde_json::json!({ "activeManifest": reduced })
}

/// The media type of an asset, for reader and ingredient
/// format hints. An image is named by the engine's own
/// sniff ([`crate::engine::sniff_content_type`]), so
/// every source the engine decodes as an image is named as
/// that image here too, with one detector for both (issue
/// #564: this module's own sniff missed the JPEG XL
/// container, most HEIF brands, SVG, TIFF, BMP and ICO).
/// Any other ISO BMFF file is a video container, since the
/// engine's sniff claims every still brand it reads, and
/// anything else (a PDF, a Matroska video) is
/// `application/octet-stream`. Content detection in the SDK
/// does the real work when reading; this gives it a
/// sensible fallback and decides which sources are
/// ingredients (see [`is_image`]).
pub fn sniff_format(bytes: &[u8]) -> &'static str {
    if let Some(mime) = crate::engine::sniff_content_type(bytes) {
        mime
    } else if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        "video/mp4"
    } else {
        "application/octet-stream"
    }
}

/// Whether a [`sniff_format`] type names an image, still
/// or animated: a source the output is transformed from,
/// which signing records as the output's parent
/// ingredient. The others are video and PDF sources, whose
/// output is a frame or a page rendered from them and
/// starts a fresh chain.
pub fn is_image(mime: &str) -> bool {
    mime.starts_with("image/")
}

/// Whether WebP bytes are an animated image (the VP8X
/// ANIM flag): signing covers stills only in v1, and an
/// animated WebP must not be fed to the signer as if it
/// were one frame.
pub fn webp_is_animated(bytes: &[u8]) -> bool {
    bytes.len() >= 21
        && &bytes[0..4] == b"RIFF"
        && &bytes[8..12] == b"WEBP"
        && &bytes[12..16] == b"VP8X"
        && bytes[20] & 0x02 != 0
}

/// Which output formats are signed: the still formats
/// proven end to end. GIF, JPEG XL, and video
/// outputs are served unsigned (meta reports the output
/// as carrying no manifest).
pub fn signable_mime(format: crate::url::ImageFormat) -> Option<&'static str> {
    match format {
        crate::url::ImageFormat::Jpeg => Some("image/jpeg"),
        crate::url::ImageFormat::Png => Some("image/png"),
        crate::url::ImageFormat::WebP => Some("image/webp"),
        crate::url::ImageFormat::Avif => Some("image/avif"),
        _ => None,
    }
}

/// The operator's signing identity: a PEM certificate
/// chain and its private key, plus an optional timestamp
/// authority. The SDK signer is built once, when
/// `from_pem` settles the algorithm: rebuilding it per
/// sign re-parsed the PEM chain and key every time, which
/// measured ~0.64 ms of a ~1.9 ms small-output sign. The
/// boxed signer is `Send + Sync`, so this struct stays
/// shareable across threads with no lock around it, and
/// signing stays a borrow.
pub struct ProvenanceSigner {
    signer: TimedSigner,
    alg: SigningAlg,
    tsa: Option<TimestampAuthority>,
    /// The identity validators will display, read back
    /// from the boot self-test's manifest (the leaf
    /// certificate's common name).
    pub name: String,
}

impl std::fmt::Debug for ProvenanceSigner {
    /// Names the identity and the algorithm only: the PEM
    /// bytes, above all the private key, never reach a
    /// debug print.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProvenanceSigner")
            .field("name", &self.name)
            .field("alg", &self.alg)
            .field("tsa", &self.tsa)
            .finish_non_exhaustive()
    }
}

/// What one signing pass records.
pub struct SignRequest<'a> {
    /// The source asset and its media type, recorded as a
    /// parent ingredient: the ingredient names the source
    /// by its format and, when the source carries a
    /// manifest store of its own, records how that store
    /// validated. The store itself never rides along (see
    /// [`ProvenanceSigner::sign`]). `None` when the source
    /// is not an image (see [`is_image`]): a video or PDF
    /// source, whose still was rendered rather than
    /// transformed.
    pub ingredient: Option<(&'a [u8], &'a str)>,
    /// The request ran an AI edit (removebg, studio,
    /// remove_region, upscale, class edits): the output's
    /// manifest records the edit with the IPTC
    /// trained-algorithm source type.
    pub ai_edited: bool,
}

impl ProvenanceSigner {
    /// Load from the operator's files and prove the
    /// identity works (see `from_pem`) before the server
    /// starts answering. A timestamp authority gets the
    /// default bounds.
    pub fn from_files(
        cert_path: &str,
        key_path: &str,
        tsa_url: Option<String>,
    ) -> Result<Self, String> {
        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| format!("cannot read CRABCROP_C2PA_CERT file {cert_path}: {e}"))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| format!("cannot read CRABCROP_C2PA_KEY file {key_path}: {e}"))?;
        Self::from_pem(cert_pem, key_pem, tsa_url.map(TimestampAuthority::new))
    }

    /// Build and validate a signing identity from PEM
    /// bytes. Validation is a real round trip: sign the
    /// self-test image and require the result to read back
    /// valid. That probe settles the algorithm (each
    /// candidate is tried until one produces a verifying
    /// signature), the certificate profile (the SDK refuses
    /// a bare self-signed certificate, an end-entity
    /// certificate without the required EKU, and any chain
    /// whose key does not match), and the timestamp
    /// authority when one is configured. A failure names
    /// the SDK's own error: boot fails fast, like the model
    /// preloads, instead of the first signed request
    /// discovering a broken identity.
    ///
    /// The algorithm search runs without the authority,
    /// and the authority is then asked exactly once, with
    /// the settled algorithm (issue #397). Probing every
    /// candidate against it would multiply a stalled
    /// authority's bound by every algorithm the key
    /// happens to load under (an RSA key loads as all
    /// three PS variants), and a failed timestamp says
    /// nothing about the algorithm anyway.
    pub fn from_pem(
        cert_pem: Vec<u8>,
        key_pem: Vec<u8>,
        tsa: Option<TimestampAuthority>,
    ) -> Result<Self, String> {
        let mut first_error = None;
        for alg in [
            SigningAlg::Es256,
            SigningAlg::Ed25519,
            SigningAlg::Es384,
            SigningAlg::Es512,
            SigningAlg::Ps256,
            SigningAlg::Ps384,
            SigningAlg::Ps512,
        ] {
            let signer = match Self::build(&cert_pem, &key_pem, alg, None) {
                Ok(signer) => signer,
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                    continue;
                }
            };
            match signer.probe() {
                Ok(name) => {
                    let Some(tsa) = tsa else {
                        return Ok(Self { name, ..signer });
                    };
                    let url = tsa.url.clone();
                    return Self::build(&cert_pem, &key_pem, alg, Some(tsa))
                        .and_then(|signer| {
                            let name = signer.probe()?;
                            Ok(Self { name, ..signer })
                        })
                        .map_err(|e| {
                            format!(
                                "the timestamp authority {url} failed the signing self-test: {e}"
                            )
                        });
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        Err(format!(
            "the configured C2PA identity cannot sign: {}",
            first_error.unwrap_or_else(|| "no signing algorithm worked".to_string())
        ))
    }

    /// The algorithm the boot probe settled on. Boot names
    /// it, and warns for an RSA key (issue #600).
    pub fn alg(&self) -> SigningAlg {
        self.alg
    }

    /// One candidate signer, not yet probed. Every SDK
    /// HTTP call signing makes goes through an agent with
    /// the authority's bounds (the defaults when there is
    /// no authority).
    fn build(
        cert_pem: &[u8],
        key_pem: &[u8],
        alg: SigningAlg,
        tsa: Option<TimestampAuthority>,
    ) -> Result<Self, String> {
        let inner = c2pa::create_signer::from_keys(
            cert_pem,
            key_pem,
            alg,
            tsa.as_ref().map(|t| t.url.clone()),
        )
        .map_err(|e| e.to_string())?;
        let agent = match &tsa {
            Some(t) => tsa_agent(t.connect_timeout, t.request_timeout),
            None => tsa_agent(TSA_CONNECT_TIMEOUT, TSA_REQUEST_TIMEOUT),
        };
        Ok(Self {
            signer: TimedSigner { inner, agent },
            alg,
            tsa,
            name: String::new(),
        })
    }

    /// The boot probe: sign the self-test image with this
    /// identity and read the manifest back. Ok carries the
    /// signer name the readback reports.
    fn probe(&self) -> Result<String, String> {
        let signed = self
            .sign(
                SELF_TEST_PNG,
                "image/png",
                &SignRequest {
                    ingredient: None,
                    ai_edited: false,
                },
            )
            .map_err(|e| e.to_string())?;
        let summary = read_manifest(&signed, "image/png")
            .ok_or_else(|| "self-test output carried no readable manifest".to_string())?;
        if summary.state == "invalid" {
            return Err("self-test signature did not validate".to_string());
        }
        summary
            .signer
            .ok_or_else(|| "self-test manifest named no signer".to_string())
    }

    /// Sign one output. The returned bytes are the output
    /// with its manifest store embedded; the input bytes
    /// are never modified. A failure of the timestamp
    /// authority is [`SignError::Authority`], anything else
    /// [`SignError::Failed`].
    ///
    /// The source is the output's parent ingredient, built
    /// here rather than read whole from the source by the
    /// SDK: it names the source's format and, when the
    /// source carries a manifest store, the status codes its
    /// validation produced ([`source_status`]), and nothing
    /// else. The source's store does not ride along, under
    /// either metadata policy. The store is the source's
    /// metadata, not the output's: its claim thumbnails
    /// depict the source as it was, and its assertions
    /// (EXIF, authorship, identity) describe the source, so
    /// carrying it could not follow `CRABCROP_METADATA` (a
    /// parent claim's title, generator and certificate are
    /// signed, so no SDK redaction removes them), and it made
    /// a thumbnail-sized output as large as the source's
    /// store whatever the request asked for (issue #610). No
    /// instance ID is copied either: the SDK would take the
    /// source's XMP `InstanceID`, which identifies the
    /// original file.
    pub fn sign(
        &self,
        output: &[u8],
        mime: &str,
        request: &SignRequest<'_>,
    ) -> Result<Vec<u8>, SignError> {
        let failed = |e: c2pa::Error| SignError::Failed(e.to_string());
        let definition = serde_json::json!({
            "title": "crabcrop output",
            "format": mime,
            "claim_generator_info": [{
                "name": "crabcrop",
                "version": env!("CARGO_PKG_VERSION"),
            }],
        })
        .to_string();
        let mut builder = Builder::from_context(self.signer.context())
            .with_definition(definition)
            .map_err(failed)?;
        // The actions assertion must open with a created
        // or opened action (the SDK's own post-sign
        // validation enforces it): opened when a parent
        // ingredient exists to open, created when the
        // output starts a fresh chain.
        if let Some((source_bytes, source_format)) = request.ingredient {
            // A bare JPEG XL codestream has no box a manifest
            // could live in, and the SDK's JPEG XL reader
            // refuses one outright instead of finding no
            // manifest (issue #564). So it is read as a type
            // no SDK reader claims, which is how a BMP or ICO
            // source (no reader at all) goes in: nothing is
            // read from it, and the ingredient still names
            // its format.
            let read_as =
                if source_format == "image/jxl" && crate::engine::is_jxl_codestream(source_bytes) {
                    "application/octet-stream"
                } else {
                    source_format
                };
            let mut ingredient = serde_json::json!({
                "title": "source",
                "format": source_format,
                "relationship": "parentOf",
                "label": "source",
            });
            if let Some(status) = source_status(source_bytes, read_as) {
                ingredient["validation_results"] = status;
            }
            let ingredient =
                c2pa::Ingredient::from_json(&ingredient.to_string()).map_err(failed)?;
            builder.add_ingredient(ingredient);
            builder
                .add_action(serde_json::json!({
                    "action": "c2pa.opened",
                    "parameters": { "ingredientIds": ["source"] },
                }))
                .map_err(failed)?;
        } else {
            // No ingredient to open: the source was a video
            // or a PDF, whose frame or page this output was
            // rendered from (every image source is an
            // ingredient). The chain starts here, with the
            // software-image source type. A created action
            // must carry a digital source type (the
            // post-sign validation enforces that too).
            builder
                .add_action(serde_json::json!({
                    "action": "c2pa.created",
                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/softwareImage",
                }))
                .map_err(failed)?;
        }
        if request.ai_edited {
            builder
                .add_action(serde_json::json!({
                    "action": "c2pa.edited",
                    "digitalSourceType": AI_EDITED_SOURCE_TYPE,
                }))
                .map_err(failed)?;
        }
        let mut source = Cursor::new(output);
        let mut dest = Cursor::new(Vec::new());
        let pass = self.signer.pass();
        builder
            .sign(&pass, mime, &mut source, &mut dest)
            .map_err(|e| {
                if pass.authority_failed.get() {
                    SignError::Authority(e.to_string())
                } else {
                    failed(e)
                }
            })?;
        Ok(dest.into_inner())
    }
}

// Test-only identity, generated for the suites: a
// throwaway CA and its end-entity certificate (ES256,
// the C2PA certificate profile: EKU emailProtection,
// digitalSignature), valid 2026-2126. It signs test
// fixtures and nothing else; it is not a deployment
// credential and chains to no trust list, so manifests
// it produces validate as valid-not-trusted, exactly
// like a private operator CA. Not `cfg(test)`: the
// binary's own tests (`src/main.rs`, a separate crate from
// this library) build a signer from it. Constants are
// inlined where they are used, so nothing reaches a
// production build.
#[doc(hidden)]
pub const TEST_CHAIN: &str = "-----BEGIN CERTIFICATE-----\nMIIB4jCCAYmgAwIBAgIUcCXdPow++l0p0FSHJK0c0zO+/IEwCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMjgxODUxMTJaGA8yMTI2MTAwNDE4NTExMlowNzEdMBsGA1UE\nAwwUY3JhYmNyb3AgdGVzdCBzaWduZXIxFjAUBgNVBAoMDUNyYWJjcm9wIFRlc3Qw\nWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQeWY39h9h9eukk2qnrXHGqU7upywvS\n2ySnXDnczKViN62hVTtNVDyjwJQq7jaU/w2+/l5uztgBmPcNBv6vEHgYo3UwczAM\nBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQEAwIHgDATBgNVHSUEDDAKBggrBgEFBQcD\nBDAdBgNVHQ4EFgQU1rrLbUlCvE/gZ99fu6wAidIIFZIwHwYDVR0jBBgwFoAUqZMw\nTa6P7Eg+tNMjaGpBUfqat0kwCgYIKoZIzj0EAwIDRwAwRAIgdVWHqXBgtrLDtisG\nYaEtvsVjh28X/X3WC0Mbz+TGeZoCIEPwQFXP02YZNTBNI4qH1YIcw4MEuXMOzRpp\nbJxSXbTw\n-----END CERTIFICATE-----\n-----BEGIN CERTIFICATE-----\nMIIBvjCCAWOgAwIBAgIUc/mA9ToPXNgspCO2gfKnOpMb8SwwCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMjgxODUxMTJaGA8yMTI2MTAwNDE4NTExMlowMzEZMBcGA1UE\nAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3AgVGVzdDBZMBMG\nByqGSM49AgEGCCqGSM49AwEHA0IABLEwHrQQMmP9/Q1BtF0JwPPjDhVH+vgbyjO2\nDOpW8ITW8JjgyxwK245BpXepHOFRX6Sf5av2uMQPdP8e+u0/DBqjUzBRMB0GA1Ud\nDgQWBBSpkzBNro/sSD600yNoakFR+pq3STAfBgNVHSMEGDAWgBSpkzBNro/sSD60\n0yNoakFR+pq3STAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMCA0kAMEYCIQDu\nxLkJML8djr0Vl20Vk3290vebT5umm6WgDgXzsUUQOQIhANsnF3kFIH5/M2hYt8uF\n87WSZPCiSXsWSaqK/HovPHh8\n-----END CERTIFICATE-----\n";
#[doc(hidden)]
pub const TEST_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgDdKjjSDmVGisl6rO\nPWFhHIxOEZSN588utCGcu+P9cUGhRANCAAQeWY39h9h9eukk2qnrXHGqU7upywvS\n2ySnXDnczKViN62hVTtNVDyjwJQq7jaU/w2+/l5uztgBmPcNBv6vEHgY\n-----END PRIVATE KEY-----\n";

/// A test-identity signer for other modules' suites.
#[cfg(test)]
pub(crate) fn test_signer() -> std::sync::Arc<ProvenanceSigner> {
    std::sync::Arc::new(tests::test_signer())
}

/// A test-identity signer whose timestamp authority is a
/// stalled one with short bounds, for other modules'
/// suites (issue #467). Built without the boot probe,
/// which the stalled authority would fail: this is the
/// signer whose authority stopped answering after boot.
/// Drop the sender to release the authority; the join
/// handle yields how many connections it accepted.
#[cfg(test)]
pub(crate) fn test_signer_with_a_stalled_authority() -> (
    std::sync::Arc<ProvenanceSigner>,
    std::sync::mpsc::Sender<()>,
    std::thread::JoinHandle<usize>,
) {
    let (url, release, staller) = tests::stalled_authority();
    let signer = ProvenanceSigner::build(
        TEST_CHAIN.as_bytes(),
        TEST_KEY.as_bytes(),
        SigningAlg::Es256,
        Some(tests::short_bounds(url)),
    )
    .expect("the test identity loads");
    (std::sync::Arc::new(signer), release, staller)
}

/// What [`test_source_with_credentials`] writes into its
/// source's XMP packet and manifest store: its file title,
/// XMP instance ID, GPS latitude, camera body serial and
/// author. A derived output must carry none of them.
#[cfg(test)]
pub(crate) const TEST_SOURCE_FACTS: [&str; 5] = [
    "IMG_TEST_0001.jpg",
    "xmp.iid:test-source-instance-0001",
    "48,51.494N",
    "SN-TEST-0042",
    "Test Author Name",
];

/// A JPEG shaped like a Content Credentials camera's export,
/// for the suites: a 200x150 frame (a red block on the left,
/// a blue one on the right), an XMP packet naming its
/// instance ID, and a manifest store signed with the test
/// identity whose claim thumbnail is the whole frame and
/// whose assertions carry EXIF location, a body serial and
/// an author ([`TEST_SOURCE_FACTS`]). `thumbnail_padding`
/// bytes follow the thumbnail's own end, to size the store.
#[cfg(test)]
pub(crate) fn test_source_with_credentials(thumbnail_padding: usize) -> Vec<u8> {
    let [title, instance_id, latitude, serial, author] = TEST_SOURCE_FACTS;
    let frame = image::RgbImage::from_fn(200, 150, |x, _| {
        if x < 100 {
            image::Rgb([220, 0, 0])
        } else {
            image::Rgb([0, 90, 200])
        }
    });
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(frame)
        .write_to(&mut encoded, image::ImageFormat::Jpeg)
        .expect("the frame encodes");
    let frame = encoded.into_inner();
    let packet = format!(
        "http://ns.adobe.com/xap/1.0/\0<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF \
         xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description \
         rdf:about=\"\" xmlns:xmpMM=\"http://ns.adobe.com/xap/1.0/mm/\" \
         xmpMM:InstanceID=\"{instance_id}\"/></rdf:RDF></x:xmpmeta>"
    );
    let segment_len = u16::try_from(packet.len() + 2).expect("the packet fits a segment");
    let mut source = frame[..2].to_vec();
    source.extend_from_slice(&[0xFF, 0xE1]);
    source.extend_from_slice(&segment_len.to_be_bytes());
    source.extend_from_slice(packet.as_bytes());
    source.extend_from_slice(&frame[2..]);
    let mut thumbnail = frame;
    thumbnail.resize(thumbnail.len() + thumbnail_padding, 0x5A);
    let definition = serde_json::json!({
        "title": title,
        "claim_generator_info": [{ "name": "test camera firmware", "version": "1.0" }],
        "assertions": [
            { "label": "c2pa.actions", "data": { "actions": [{
                "action": "c2pa.created",
                "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
            }]}},
            { "label": "stds.exif", "data": {
                "@context": { "exif": "http://ns.adobe.com/exif/1.0/" },
                "exif:GPSLatitude": latitude,
                "exif:BodySerialNumber": serial,
            }},
            { "label": "stds.schema-org.CreativeWork", "data": {
                "@context": "https://schema.org",
                "@type": "CreativeWork",
                "author": [{ "@type": "Person", "name": author }],
            }},
        ],
    })
    .to_string();
    let mut builder = Builder::from_context(Context::new())
        .with_definition(definition)
        .expect("the definition loads");
    builder
        .set_thumbnail("image/jpeg", &mut Cursor::new(thumbnail))
        .expect("the thumbnail is accepted");
    let signer = c2pa::create_signer::from_keys(
        TEST_CHAIN.as_bytes(),
        TEST_KEY.as_bytes(),
        SigningAlg::Es256,
        None,
    )
    .expect("the test identity loads");
    let mut signed = Cursor::new(Vec::new());
    builder
        .sign(
            &*signer,
            "image/jpeg",
            &mut Cursor::new(source),
            &mut signed,
        )
        .expect("the source signs");
    signed.into_inner()
}

/// The manifest store JSON of an asset, for other
/// modules' suites asserting on manifest contents.
#[cfg(test)]
pub(crate) fn manifest_json(bytes: &[u8], format: &str) -> String {
    let mut cursor = Cursor::new(bytes);
    Reader::from_context(Context::new())
        .with_stream(format, &mut cursor)
        .map(|r| r.json())
        .unwrap_or_default()
}

#[cfg(test)]
mod state_tests;

#[cfg(test)]
mod identity_tests;

/// The test identity whose common name runs past 64
/// characters over two lines, for other modules' suites
/// (issue #595).
#[cfg(test)]
pub(crate) fn test_signer_with_a_long_name() -> std::sync::Arc<ProvenanceSigner> {
    std::sync::Arc::new(identity_tests::long_name_signer())
}

#[cfg(test)]
mod tests {
    use super::*;
    // The test CA's key: a well-formed EC key that does not
    // match the signing certificate.
    const TEST_WRONG_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgx0ypMwM5/DPsx5jF\nWzZ5Lupqh3K8e03pDD/9clnFcdahRANCAASxMB60EDJj/f0NQbRdCcDz4w4VR/r4\nG8oztgzqVvCE1vCY4MscCtuOQaV3qRzhUV+kn+Wr9rjED3T/HvrtPwwa\n-----END PRIVATE KEY-----\n";
    // The signing key's certificate self-issued: the SDK
    // refuses a bare self-signed certificate at sign time.
    const TEST_SELF_SIGNED: &str = "-----BEGIN CERTIFICATE-----\nMIIBxTCCAWugAwIBAgIUKNFA2Y0pJgpMPwkzGmHc06DKVI0wCgYIKoZIzj0EAwIw\nNzEdMBsGA1UEAwwUY3JhYmNyb3AgdGVzdCBzaWduZXIxFjAUBgNVBAoMDUNyYWJj\ncm9wIFRlc3QwIBcNMjYxMDI4MTg1MTIxWhgPMjEyNjEwMDQxODUxMjFaMDcxHTAb\nBgNVBAMMFGNyYWJjcm9wIHRlc3Qgc2lnbmVyMRYwFAYDVQQKDA1DcmFiY3JvcCBU\nZXN0MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEHlmN/YfYfXrpJNqp61xxqlO7\nqcsL0tskp1w53MylYjetoVU7TVQ8o8CUKu42lP8Nvv5ebs7YAZj3DQb+rxB4GKNT\nMFEwHQYDVR0OBBYEFNa6y21JQrxP4GffX7usAInSCBWSMB8GA1UdIwQYMBaAFNa6\ny21JQrxP4GffX7usAInSCBWSMA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZIzj0EAwID\nSAAwRQIgWk0XBTIIqO+5HB2w3+Qjyxf/JgOqpPO7ZiISg8hXkFwCIQCi/kJ2AMxe\nwibcj2fQu05d3cJkhNy5JA4JsEoI6wwhIw==\n-----END CERTIFICATE-----\n";

    pub(crate) fn test_signer() -> ProvenanceSigner {
        ProvenanceSigner::from_pem(
            TEST_CHAIN.as_bytes().to_vec(),
            TEST_KEY.as_bytes().to_vec(),
            None,
        )
        .expect("the test identity signs")
    }

    fn plain() -> SignRequest<'static> {
        SignRequest {
            ingredient: None,
            ai_edited: false,
        }
    }

    #[test]
    fn identity_loads_and_names_itself() {
        let signer = test_signer();
        assert_eq!(signer.name, "crabcrop test signer");
    }

    #[test]
    fn sign_and_read_round_trip() {
        let signer = test_signer();
        let signed = signer
            .sign(SELF_TEST_PNG, "image/png", &plain())
            .expect("signing works");
        assert!(signed.len() > SELF_TEST_PNG.len());
        let summary = read_manifest(&signed, "image/png").expect("the output carries a manifest");
        assert_eq!(summary.state, "valid");
        assert_eq!(summary.signer.as_deref(), Some("crabcrop test signer"));
        assert_eq!(summary.manifests, 1);
        assert_eq!(summary.ingredients, 0);
    }

    #[test]
    fn unsigned_and_garbage_assets_have_no_manifest() {
        assert!(read_manifest(SELF_TEST_PNG, "image/png").is_none());
        assert!(read_manifest(b"not an image at all", "image/png").is_none());
        assert!(read_manifest(&[], "image/png").is_none());
    }

    /// A small JPEG to sign as an output: the frame a crop
    /// of the test source would keep (the blue block).
    fn crop_output() -> Vec<u8> {
        let crop = image::RgbImage::from_pixel(40, 30, image::Rgb([0, 90, 200]));
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(crop)
            .write_to(&mut encoded, image::ImageFormat::Jpeg)
            .expect("the crop encodes");
        encoded.into_inner()
    }

    fn sign_from(signer: &ProvenanceSigner, output: &[u8], mime: &str, source: &[u8]) -> Vec<u8> {
        signer
            .sign(
                output,
                mime,
                &SignRequest {
                    ingredient: Some((source, sniff_format(source))),
                    ai_edited: false,
                },
            )
            .expect("the output signs")
    }

    fn reader(signed: &[u8], mime: &str) -> Reader {
        Reader::from_context(Context::new())
            .with_stream(mime, Cursor::new(signed))
            .expect("the reader parses the output")
    }

    /// The codes one category of the parent ingredient's
    /// recorded validation status holds.
    fn parent_codes(signed: &[u8], mime: &str, category: &str) -> Vec<String> {
        let reader = reader(signed, mime);
        let parent = &reader
            .active_manifest()
            .expect("an active manifest")
            .ingredients()[0];
        let results = serde_json::to_value(parent.validation_results().expect("a status"))
            .expect("the status serializes");
        results["activeManifest"][category]
            .as_array()
            .expect("the category is a list")
            .iter()
            .map(|status| {
                assert!(status.get("url").is_none(), "no URL is recorded: {status}");
                status["code"].as_str().expect("a code").to_string()
            })
            .collect()
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn a_signed_output_names_its_parent_without_carrying_the_parent_store() {
        let signer = test_signer();
        let source = test_source_with_credentials(0);
        let output = crop_output();
        let signed = sign_from(&signer, &output, "image/jpeg", &source);
        let reader = reader(&signed, "image/jpeg");
        assert_eq!(reader.validation_state(), ValidationState::Valid);
        assert_eq!(
            reader.manifests().len(),
            1,
            "the source's store does not ride along"
        );
        let manifest = reader.active_manifest().expect("an active manifest");
        assert!(manifest.thumbnail_ref().is_none());
        let [parent] = manifest.ingredients() else {
            panic!("one ingredient: {:?}", manifest.ingredients());
        };
        assert!(parent.is_parent());
        assert_eq!(parent.format(), Some("image/jpeg"));
        assert!(parent.thumbnail_ref().is_none(), "no picture of the source");
        assert!(parent.active_manifest().is_none());
        assert!(parent.manifest_data_ref().is_none());
        assert_eq!(parent.instance_id(), "None", "no instance ID is copied");
        // The source validated, and the output says so.
        let success = parent_codes(&signed, "image/jpeg", "success");
        assert!(
            success.iter().any(|c| c == "claimSignature.validated"),
            "{success:?}"
        );
        // Nothing the source carried is in the bytes: not
        // its title, XMP instance ID, location, serial or
        // author, and no picture but the output itself.
        for fact in TEST_SOURCE_FACTS {
            assert!(!contains(&signed, fact.as_bytes()), "{fact} is not carried");
        }
        let source_label = read_label(&source);
        assert!(!contains(&signed, source_label.as_bytes()));
        let pictures = signed
            .windows(3)
            .filter(|w| *w == [0xFF, 0xD8, 0xFF])
            .count();
        assert_eq!(pictures, 1, "the output is the only picture in the bytes");
    }

    fn read_label(source: &[u8]) -> String {
        reader(source, "image/jpeg")
            .active_label()
            .expect("the source has an active manifest")
            .to_string()
    }

    #[test]
    fn the_parent_store_does_not_size_the_output() {
        let signer = test_signer();
        let output = crop_output();
        let bare = sign_from(&signer, &output, "image/jpeg", &crop_output());
        // A store under the read bound is read (the status
        // is recorded) and one over it is not, and neither
        // grows the output past a source with no store at
        // all, beyond the few status codes.
        for (padding, read) in [
            (MAX_SOURCE_STORE_BYTES / 2, true),
            (MAX_SOURCE_STORE_BYTES + 64 * 1024, false),
        ] {
            let source = test_source_with_credentials(padding);
            assert!(source.len() > padding);
            let signed = sign_from(&signer, &output, "image/jpeg", &source);
            assert!(
                signed.len() < bare.len() + 2048,
                "a {}-byte source store made a {}-byte output ({} without a store)",
                source.len(),
                signed.len(),
                bare.len()
            );
            let reader = reader(&signed, "image/jpeg");
            assert_eq!(reader.validation_state(), ValidationState::Valid);
            let parent = &reader
                .active_manifest()
                .expect("an active manifest")
                .ingredients()[0];
            assert_eq!(
                parent.validation_results().is_some(),
                read,
                "padding {padding}"
            );
        }
    }

    #[test]
    fn a_resign_records_how_the_parent_validated() {
        let signer = test_signer();
        let parent = signer
            .sign(SELF_TEST_PNG, "image/png", &plain())
            .expect("parent signs");
        let child = sign_from(&signer, SELF_TEST_PNG, "image/png", &parent);
        let summary = read_manifest(&child, "image/png").expect("the child carries a manifest");
        assert_eq!(summary.state, "valid");
        assert_eq!(summary.manifests, 1);
        assert_eq!(summary.ingredients, 1);
        let success = parent_codes(&child, "image/png", "success");
        assert!(
            success.iter().any(|c| c == "claimSignature.validated"),
            "{success:?}"
        );
        // The test CA is on no trust list: the only failure
        // is the untrusted credential a valid manifest may
        // carry.
        assert_eq!(
            parent_codes(&child, "image/png", "failure"),
            ["signingCredential.untrusted"]
        );
    }

    #[test]
    fn a_tampered_parent_is_recorded_as_failing() {
        let signer = test_signer();
        let mut parent = signer
            .sign(SELF_TEST_PNG, "image/png", &plain())
            .expect("parent signs");
        parent.push(0);
        let child = sign_from(&signer, SELF_TEST_PNG, "image/png", &parent);
        // The child's own manifest is intact; what it says
        // about its parent is that the parent's hash failed.
        assert_eq!(
            reader(&child, "image/png").validation_state(),
            ValidationState::Valid
        );
        let failure = parent_codes(&child, "image/png", "failure");
        assert!(
            failure.iter().any(|c| c == "assertion.dataHash.mismatch"),
            "{failure:?}"
        );
    }

    #[test]
    fn a_source_store_that_does_not_parse_does_not_fail_signing() {
        // Break the store's outer JUMBF box: the store is
        // still there to find, but no reader parses it.
        let mut source = test_source_with_credentials(0);
        let at = source
            .windows(4)
            .position(|w| w == b"jumb")
            .expect("the source carries a JUMBF box");
        source[at + 2] ^= 0xFF;
        assert!(read_manifest(&source, "image/jpeg").is_none());
        let signed = sign_from(&test_signer(), &crop_output(), "image/jpeg", &source);
        let reader = reader(&signed, "image/jpeg");
        assert_eq!(reader.validation_state(), ValidationState::Valid);
        let parent = &reader
            .active_manifest()
            .expect("an active manifest")
            .ingredients()[0];
        assert_eq!(parent.format(), Some("image/jpeg"));
        assert!(parent.validation_results().is_none());
    }

    /// A tagged COSE_Sign1 whose protected header names
    /// EdDSA and the `application/vc` content type, carrying
    /// `payload` and a placeholder signature: the envelope an
    /// identity-claims-aggregation credential travels in.
    fn cose_sign1(payload: &[u8]) -> Vec<u8> {
        fn byte_string(out: &mut Vec<u8>, bytes: &[u8]) {
            match bytes.len() {
                len if len < 24 => out.push(0x40 | len as u8),
                len if len < 256 => out.extend_from_slice(&[0x58, len as u8]),
                len => {
                    out.push(0x59);
                    let len = u16::try_from(len).expect("the payload fits two length bytes");
                    out.extend_from_slice(&len.to_be_bytes());
                }
            }
            out.extend_from_slice(bytes);
        }
        // { 1 (alg): -8 (EdDSA), 3 (content type): "application/vc" }
        let mut protected = vec![0xA2, 0x01, 0x27, 0x03, 0x6E];
        protected.extend_from_slice(b"application/vc");
        // Tag 18 (COSE_Sign1), an array of four.
        let mut out = vec![0xD2, 0x84];
        byte_string(&mut out, &protected);
        out.push(0xA0);
        byte_string(&mut out, payload);
        byte_string(&mut out, &[0; 64]);
        out
    }

    /// A credential holder claiming an identity whose
    /// credential was issued by the `did:web` DID it holds.
    /// Validating such an identity means resolving the
    /// issuer's DID document from the issuer's host.
    struct WebIssuedIdentity(String);

    impl c2pa::identity::builder::CredentialHolder for WebIssuedIdentity {
        fn sig_type(&self) -> &'static str {
            "cawg.identity_claims_aggregation"
        }

        fn reserve_size(&self) -> usize {
            4096
        }

        fn sign(
            &self,
            payload: &c2pa::identity::SignerPayload,
        ) -> Result<Vec<u8>, c2pa::identity::builder::IdentityBuilderError> {
            let credential = serde_json::json!({
                "@context": [
                    "https://www.w3.org/ns/credentials/v2",
                    "https://cawg.io/identity/1.1/ica/context/",
                ],
                "type": ["VerifiableCredential", "IdentityClaimsAggregationCredential"],
                "credentialSubject": {
                    "verifiedIdentities": [{
                        "type": "cawg.document_verification",
                        "name": "Test Identity",
                        "verifiedAt": "2026-01-01T00:00:00Z",
                        "provider": { "id": "https://identity.example", "name": "Test Provider" },
                    }],
                    "c2paAsset": payload,
                },
                "issuer": self.0,
                "validFrom": "2026-01-01T00:00:00Z",
            })
            .to_string();
            Ok(cose_sign1(credential.as_bytes()))
        }
    }

    /// The test identity, adding a [`WebIssuedIdentity`]
    /// identity assertion to every manifest it signs.
    struct SignerWithIdentity {
        inner: c2pa::BoxedSigner,
        issuer: String,
    }

    impl Signer for SignerWithIdentity {
        fn sign(&self, data: &[u8]) -> c2pa::Result<Vec<u8>> {
            self.inner.sign(data)
        }

        fn alg(&self) -> SigningAlg {
            self.inner.alg()
        }

        fn certs(&self) -> c2pa::Result<Vec<Vec<u8>>> {
            self.inner.certs()
        }

        fn reserve_size(&self) -> usize {
            self.inner.reserve_size()
        }

        fn dynamic_assertions(&self) -> Vec<Box<dyn c2pa::dynamic_assertion::DynamicAssertion>> {
            vec![Box::new(
                c2pa::identity::builder::IdentityAssertionBuilder::for_credential_holder(
                    WebIssuedIdentity(self.issuer.clone()),
                ),
            )]
        }
    }

    /// A JPEG whose manifest store, signed with the test
    /// identity, carries a CAWG identity assertion issued by
    /// `issuer`.
    fn test_source_with_a_web_identity(issuer: &str) -> Vec<u8> {
        let signer = SignerWithIdentity {
            inner: c2pa::create_signer::from_keys(
                TEST_CHAIN.as_bytes(),
                TEST_KEY.as_bytes(),
                SigningAlg::Es256,
                None,
            )
            .expect("the test identity loads"),
            issuer: issuer.to_string(),
        };
        let definition = serde_json::json!({
            "title": "identity test source",
            "claim_generator_info": [{ "name": "test editor", "version": "1.0" }],
            "assertions": [{ "label": "c2pa.actions", "data": { "actions": [{
                "action": "c2pa.created",
                "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
            }]}}],
        })
        .to_string();
        let mut signed = Cursor::new(Vec::new());
        Builder::from_context(reading_context())
            .with_definition(definition)
            .expect("the definition loads")
            .sign(
                &signer,
                "image/jpeg",
                &mut Cursor::new(crop_output()),
                &mut signed,
            )
            .expect("the source signs");
        signed.into_inner()
    }

    #[test]
    fn a_source_identity_assertion_is_not_validated_while_signing() {
        // The issuer's host is one the SDK refuses to fetch
        // from before any request is made, so this test opens
        // no connection whichever way signing reads the
        // source. Validating the identity would still record
        // CAWG status codes for the parent; reading the
        // source's store without decoding identities records
        // none, and needs nothing but local work.
        let source = test_source_with_a_web_identity("did:web:localhost%3A9:test-issuer");
        assert!(contains(&source, b"cawg.identity"));
        assert!(contains(&source, b"did:web:localhost%3A9:test-issuer"));
        let signed = sign_from(&test_signer(), &crop_output(), "image/jpeg", &source);
        assert_eq!(
            reader(&signed, "image/jpeg").validation_state(),
            ValidationState::Valid
        );
        let mut recorded = Vec::new();
        for category in ["success", "informational", "failure"] {
            recorded.extend(parent_codes(&signed, "image/jpeg", category));
        }
        assert!(
            recorded.iter().any(|c| c == "claimSignature.validated"),
            "the source's store was read: {recorded:?}"
        );
        assert!(
            recorded.iter().all(|c| !c.starts_with("cawg.")),
            "no identity was validated: {recorded:?}"
        );
        // Meta's summary of the source is its manifest's own
        // signature and hashes, which validate, whether or not
        // the identity's issuer could be reached.
        let summary = read_manifest(&source, "image/jpeg").expect("the source has a manifest");
        assert_eq!(summary.state, "valid");
        assert_eq!(summary.signer.as_deref(), Some("crabcrop test signer"));
        assert_eq!(summary.manifests, 1);
    }

    /// A local listener that accepts and drops connections,
    /// counting them. Send on the sender to stop it; the join
    /// handle yields the count.
    fn connection_counter() -> (
        std::net::SocketAddr,
        std::sync::mpsc::Sender<()>,
        std::thread::JoinHandle<usize>,
    ) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        let counter = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            let mut accepted = 0;
            while std::time::Instant::now() < deadline
                && stopped.try_recv() == Err(std::sync::mpsc::TryRecvError::Empty)
            {
                match listener.accept() {
                    Ok(_) => accepted += 1,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
            while listener.accept().is_ok() {
                accepted += 1;
            }
            accepted
        });
        (addr, stop, counter)
    }

    #[test]
    fn reading_a_manifest_store_opens_no_connection() {
        // Every store read (the parent's status, meta's
        // summaries) runs under this context: a request the
        // store's contents lead the SDK to make is answered
        // with an error on the spot, with no connection, and
        // identity assertions are not decoded at all.
        let context = reading_context();
        let (addr, stop, counter) = connection_counter();
        let request = c2pa::http::http::Request::get(format!("http://{addr}/did.json"))
            .body(Vec::new())
            .unwrap();
        let started = std::time::Instant::now();
        let response = context.resolver().http_resolve(request);
        assert!(response.is_err(), "the request is refused");
        assert!(started.elapsed() < Duration::from_secs(1));
        stop.send(()).unwrap();
        assert_eq!(counter.join().unwrap(), 0, "no connection was opened");
        assert!(!context.settings().core.decode_identity_assertions);
    }

    #[test]
    fn the_store_validated_is_the_store_measured() {
        // A text SVG whose leading comment puts `ftyp` at
        // bytes 4 to 7, where an ISO BMFF file names its
        // first box. The engine's sniff and the store's
        // measurement read it as SVG; the reader is handed
        // that measured store, so it validates it against
        // the SVG, instead of finding a store again through
        // its own container detection.
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60"><rect width="100" height="60" fill="#00ff00"/></svg>"##;
        let signer = test_signer();
        let mut source = b"<!--ftyp-->".to_vec();
        source.extend(
            signer
                .sign(svg, "image/svg+xml", &plain())
                .expect("the SVG signs"),
        );
        assert_eq!(&source[4..8], b"ftyp");
        assert_eq!(sniff_format(&source), "image/svg+xml");
        let signed = sign_from(&signer, &crop_output(), "image/jpeg", &source);
        let reader = reader(&signed, "image/jpeg");
        let parent = &reader
            .active_manifest()
            .expect("an active manifest")
            .ingredients()[0];
        assert_eq!(parent.format(), Some("image/svg+xml"));
        assert!(
            parent.validation_results().is_some(),
            "the measured store's status is recorded"
        );
        assert!(
            parent_codes(&signed, "image/jpeg", "success")
                .iter()
                .any(|c| c == "claimSignature.validated"),
            "the store's own signature validated"
        );
    }

    #[test]
    fn the_recorded_status_keeps_distinct_plain_codes_within_bounds() {
        let many: Vec<serde_json::Value> = (0..100)
            .map(
                |i| serde_json::json!({ "code": format!("code.{i}"), "url": "self#jumbf=/c2pa/x" }),
            )
            .collect();
        let results = serde_json::json!({
            "activeManifest": {
                "success": many,
                "informational": [
                    { "code": "signingCredential.ocsp.skipped", "explanation": "skipped" },
                    { "code": "signingCredential.ocsp.skipped" },
                    { "code": "not a code" },
                    { "code": "x".repeat(MAX_STATUS_CODE_LEN + 1) },
                    { "code": 7 },
                ],
                "failure": [],
            },
            "ingredientDeltas": [{
                "ingredientAssertionURI": "self#jumbf=/c2pa/x/c2pa.assertions/c2pa.ingredient.v3",
                "validationDeltas": {
                    "success": [],
                    "informational": [],
                    "failure": [{ "code": "assertion.dataHash.mismatch", "url": "u" }],
                },
            }],
        });
        let reduced = status_codes(&results);
        let codes = |category: &str| -> Vec<serde_json::Value> {
            reduced["activeManifest"][category]
                .as_array()
                .unwrap()
                .clone()
        };
        assert_eq!(codes("success").len(), MAX_STATUS_CODES);
        assert_eq!(codes("success")[0], serde_json::json!({ "code": "code.0" }));
        assert_eq!(
            codes("informational"),
            [serde_json::json!({ "code": "signingCredential.ocsp.skipped" })]
        );
        assert_eq!(
            codes("failure"),
            [serde_json::json!({ "code": "assertion.dataHash.mismatch" })]
        );
        assert!(reduced.get("ingredientDeltas").is_none());
    }

    #[test]
    fn ai_edited_outputs_carry_the_source_type() {
        let signer = test_signer();
        let signed = signer
            .sign(
                SELF_TEST_PNG,
                "image/png",
                &SignRequest {
                    ingredient: None,
                    ai_edited: true,
                },
            )
            .expect("signing works");
        let summary = read_manifest(&signed, "image/png").expect("the output carries a manifest");
        assert_eq!(summary.state, "valid");
        let mut cursor = Cursor::new(&signed);
        let reader = Reader::from_context(Context::new())
            .with_stream("image/png", &mut cursor)
            .expect("reader parses the output");
        let json = reader.json();
        assert!(
            json.contains("compositeWithTrainedAlgorithmicMedia"),
            "the edit action records the IPTC source type: {json}"
        );
    }

    #[test]
    fn tampered_output_reads_invalid_not_absent() {
        let signer = test_signer();
        let mut signed = signer
            .sign(SELF_TEST_PNG, "image/png", &plain())
            .expect("signing works");
        // Append a byte: the manifest still parses (its
        // chunk is untouched), but the asset hash the
        // claim signed no longer matches the bytes.
        signed.push(0);
        let summary = read_manifest(&signed, "image/png").expect("the manifest still parses");
        assert_eq!(summary.state, "invalid");
    }

    #[test]
    fn self_signed_identity_is_refused_at_load() {
        let err = match ProvenanceSigner::from_pem(
            TEST_SELF_SIGNED.as_bytes().to_vec(),
            TEST_KEY.as_bytes().to_vec(),
            None,
        ) {
            Ok(_) => panic!("a bare self-signed certificate cannot sign"),
            Err(e) => e,
        };
        assert!(
            err.contains("self-signed"),
            "the SDK's refusal is surfaced: {err}"
        );
    }

    #[test]
    fn mismatched_key_is_refused_at_load() {
        let err = match ProvenanceSigner::from_pem(
            TEST_CHAIN.as_bytes().to_vec(),
            TEST_WRONG_KEY.as_bytes().to_vec(),
            None,
        ) {
            Ok(_) => panic!("a key that does not match the certificate cannot sign"),
            Err(e) => e,
        };
        assert!(!err.is_empty());
    }

    /// A timestamp authority that accepts every
    /// connection and never answers. It holds each
    /// connection until `release` is dropped, or for 20
    /// seconds at most, so a regression fails its test
    /// instead of hanging the suite. The join handle
    /// yields how many connections it accepted.
    pub(crate) fn stalled_authority() -> (
        String,
        std::sync::mpsc::Sender<()>,
        std::thread::JoinHandle<usize>,
    ) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let (release, released) = std::sync::mpsc::channel::<()>();
        let staller = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            let mut held = Vec::new();
            while std::time::Instant::now() < deadline
                && released.try_recv() == Err(std::sync::mpsc::TryRecvError::Empty)
            {
                match listener.accept() {
                    Ok((stream, _)) => held.push(stream),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
            // Count a connection still queued when the
            // release came.
            while let Ok((stream, _)) = listener.accept() {
                held.push(stream);
            }
            held.len()
        });
        (url, release, staller)
    }

    /// Short bounds for the stalled-authority tests.
    pub(crate) fn short_bounds(url: String) -> TimestampAuthority {
        TimestampAuthority {
            url,
            connect_timeout: Duration::from_millis(300),
            request_timeout: Duration::from_millis(500),
        }
    }

    #[test]
    fn boot_probe_gives_up_on_a_stalled_authority() {
        // Issue #397: the timestamp request used the SDK's
        // trait default, which builds its own untimed
        // context, so a stalled authority hung boot with no
        // limit. Through the real constructor, the probe
        // must fail at the configured bound, after asking
        // the authority exactly once.
        let (url, release, staller) = stalled_authority();
        let tsa = short_bounds(url.clone());
        let bound = tsa.request_timeout;
        let started = std::time::Instant::now();
        let result = ProvenanceSigner::from_pem(
            TEST_CHAIN.as_bytes().to_vec(),
            TEST_KEY.as_bytes().to_vec(),
            Some(tsa),
        );
        let elapsed = started.elapsed();
        drop(release);
        let contacts = staller.join().unwrap();
        let err = match result {
            Ok(_) => panic!("a stalled authority cannot timestamp the self-test"),
            Err(e) => e,
        };
        assert!(
            elapsed < Duration::from_secs(5),
            "boot gave up at the bound, took {elapsed:?}"
        );
        assert!(
            elapsed >= bound,
            "the failure came from the bound, not an earlier error: {elapsed:?}, {err}"
        );
        assert_eq!(contacts, 1, "the authority is asked once: {err}");
        assert!(err.contains(&url), "the error names the authority: {err}");
    }

    #[test]
    fn signing_gives_up_on_an_authority_that_stalls_after_boot() {
        // The per-request path: a signer whose authority
        // stops answering after boot fails each signed
        // output at the bound instead of holding the
        // process job.
        let (url, release, staller) = stalled_authority();
        let tsa = short_bounds(url);
        let bound = tsa.request_timeout;
        let signer = ProvenanceSigner::build(
            TEST_CHAIN.as_bytes(),
            TEST_KEY.as_bytes(),
            SigningAlg::Es256,
            Some(tsa),
        )
        .expect("the test identity loads");
        let started = std::time::Instant::now();
        let result = signer.sign(SELF_TEST_PNG, "image/png", &plain());
        let elapsed = started.elapsed();
        drop(release);
        let contacts = staller.join().unwrap();
        // Issue #467: the failure says it was the
        // authority, so the server can answer an outage
        // worth retrying instead of a broken encoder.
        assert!(
            matches!(result, Err(SignError::Authority(_))),
            "a stalled authority is an authority failure: {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "signing gave up at the bound, took {elapsed:?}"
        );
        assert!(elapsed >= bound, "{elapsed:?}");
        assert_eq!(contacts, 1);
    }

    #[test]
    fn an_authority_that_refuses_the_connection_is_an_authority_failure() {
        // An authority that is down outright (nothing
        // listens on its port) fails at once, and is the
        // same kind of failure as a stalled one.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        drop(listener);
        let signer = ProvenanceSigner::build(
            TEST_CHAIN.as_bytes(),
            TEST_KEY.as_bytes(),
            SigningAlg::Es256,
            Some(short_bounds(url)),
        )
        .expect("the test identity loads");
        let result = signer.sign(SELF_TEST_PNG, "image/png", &plain());
        assert!(matches!(result, Err(SignError::Authority(_))), "{result:?}");
    }

    #[test]
    fn a_failure_before_the_timestamp_is_not_the_authoritys() {
        // Issue #467: an authority being configured does
        // not make every failure its fault. Bytes that are
        // not the PNG they claim to be fail before the
        // claim is signed, so the authority is never asked
        // and the failure stays an ordinary one. Asserted
        // through the stalled authority's contact count.
        let (url, release, staller) = stalled_authority();
        let signer = ProvenanceSigner::build(
            TEST_CHAIN.as_bytes(),
            TEST_KEY.as_bytes(),
            SigningAlg::Es256,
            Some(short_bounds(url)),
        )
        .expect("the test identity loads");
        let result = signer.sign(b"not a png at all", "image/png", &plain());
        drop(release);
        let contacts = staller.join().unwrap();
        assert!(matches!(result, Err(SignError::Failed(_))), "{result:?}");
        assert_eq!(contacts, 0, "the authority was never asked");
    }

    /// A timestamp authority that answers its first
    /// request with a 307 to `location`, then exits. The
    /// join handle yields whether it served a request
    /// (it gives up after 20 seconds without one).
    fn redirecting_authority(location: String) -> (String, std::thread::JoinHandle<bool>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let redirector = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            && std::time::Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => return false,
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            // Read the whole request before answering, so
            // closing the socket never resets it mid-send.
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                    let length = head
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => return false,
                    Ok(n) => request.extend_from_slice(&chunk[..n]),
                }
            }
            write!(
                stream,
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: {location}\r\n\
                 Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .is_ok()
        });
        (url, redirector)
    }

    #[test]
    fn a_redirecting_authority_is_not_followed() {
        // The timestamp request goes through the timed
        // agent, which follows no redirect: a timestamp
        // reply is a direct answer, and one hop keeps the
        // bound on the whole call. The SDK's default stack,
        // which this path no longer uses, followed up to
        // 10 hops. A redirecting authority fails boot with
        // its status named instead.
        let (target, release, staller) = stalled_authority();
        let (url, redirector) = redirecting_authority(target);
        let result = ProvenanceSigner::from_pem(
            TEST_CHAIN.as_bytes().to_vec(),
            TEST_KEY.as_bytes().to_vec(),
            Some(short_bounds(url.clone())),
        );
        drop(release);
        let followed = staller.join().unwrap();
        assert!(redirector.join().unwrap(), "the authority was asked");
        let err = match result {
            Ok(_) => panic!("a redirect is not a timestamp"),
            Err(e) => e,
        };
        assert_eq!(followed, 0, "the redirect was not followed: {err}");
        assert!(err.contains("status = 307"), "the status is named: {err}");
        assert!(err.contains(&url), "the error names the authority: {err}");
    }

    #[test]
    fn authority_defaults_match_the_documented_bounds() {
        // docs/meta.md and docs/configuration.md promise
        // 5 seconds to connect and 15 overall.
        let tsa = TimestampAuthority::new("http://tsa.example/".to_string());
        assert_eq!(tsa.connect_timeout, Duration::from_secs(5));
        assert_eq!(tsa.request_timeout, Duration::from_secs(15));
    }

    #[test]
    fn webp_animation_flag_is_read_from_the_vp8x_header() {
        let mut still = b"RIFF\x00\x00\x00\x00WEBPVP8X".to_vec();
        // The VP8X chunk size (4 bytes) and the flags byte
        // that follows it at offset 20.
        still.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00]);
        assert!(!webp_is_animated(&still));
        let mut animated = still.clone();
        animated[20] = 0x02;
        assert!(webp_is_animated(&animated));
        assert!(!webp_is_animated(b"RIFF"));
    }
}
