//! C2PA provenance (issue #173): verify the Content
//! Credentials a source carries, and sign outputs when the
//! operator configured a signing identity.
//!
//! The design:
//!
//! - A manifest never survives this pipeline byte-wise:
//!   decode and re-encode drop the JPEG APP11 / PNG chunk a
//!   manifest lives in, in both metadata modes (proven
//!   empirically). "Preserve" therefore means re-signing the
//!   output with the source's manifest embedded as a parent
//!   ingredient, which keeps the chain intact and is the
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
    /// Validation state: `valid`, `trusted`, or `invalid`.
    pub state: &'static str,
    /// The active manifest's signer, as a validator would
    /// show it: the certificate's common name, or its
    /// issuer when the name is absent.
    pub signer: Option<String>,
    /// Manifests in the store (a re-signed output carries
    /// its ingredients' stores too).
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

/// Read and validate the manifest store embedded in an
/// asset. `None` when the asset carries no manifest (or no
/// parseable one); a present-but-failing manifest reads
/// back with state `invalid`, never as absent. `format` is
/// a MIME type hint: the reader detects the container from
/// the bytes first and uses the hint only as a fallback.
pub fn read_manifest(bytes: &[u8], format: &str) -> Option<ManifestSummary> {
    let mut cursor = Cursor::new(bytes);
    let reader = Reader::from_context(Context::new())
        .with_stream(format, &mut cursor)
        .ok()?;
    let state = match reader.validation_state() {
        ValidationState::Valid => "valid",
        ValidationState::Trusted => "trusted",
        ValidationState::Invalid => "invalid",
    };
    let active = reader.active_manifest();
    let signer = active
        .and_then(|m| m.signature_info())
        .and_then(|info| info.common_name.clone().or_else(|| info.issuer.clone()));
    Some(ManifestSummary {
        state,
        signer,
        manifests: reader.manifests().len(),
        ingredients: active.map(|m| m.ingredients().len()).unwrap_or(0),
    })
}

/// Best-effort container sniff for reader and ingredient
/// format hints. Content detection in the SDK does the
/// real work; this only gives it a sensible fallback.
pub fn sniff_format(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        "image/jpeg"
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        let brand = &bytes[8..12];
        if brand == b"avif" || brand == b"avis" {
            "image/avif"
        } else if brand == b"heic" || brand == b"heix" {
            "image/heic"
        } else {
            "video/mp4"
        }
    } else if bytes.starts_with(b"GIF8") {
        "image/gif"
    } else if bytes.starts_with(&[0xff, 0x0a]) || bytes.starts_with(b"\x00\x00\x00\x0c jxl") {
        "image/jxl"
    } else {
        "application/octet-stream"
    }
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
    /// The source asset, recorded as a parent
    /// ingredient: its bytes are hashed into the output's
    /// manifest, and when the source carries a manifest
    /// store of its own, that store rides along so the
    /// chain stays walkable. `None` when the source is
    /// not an asset an ingredient can be taken from (a
    /// video or PDF source, whose still was rendered
    /// rather than transformed).
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
            let ingredient = serde_json::json!({
                "title": "source",
                "relationship": "parentOf",
                "label": "source",
            })
            .to_string();
            let mut cursor = Cursor::new(source_bytes);
            builder
                .add_ingredient_from_stream(ingredient, source_format, &mut cursor)
                .map_err(failed)?;
            builder
                .add_action(serde_json::json!({
                    "action": "c2pa.opened",
                    "parameters": { "ingredientIds": ["source"] },
                }))
                .map_err(failed)?;
        } else {
            // No ingredient to open (the source was a
            // video or a PDF, whose still this output was
            // rendered from): the chain starts here, with
            // the software-image source type. A created
            // action must carry a digital source type (the
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

    #[test]
    fn ingredient_chain_survives_a_resign() {
        let signer = test_signer();
        let parent = signer
            .sign(SELF_TEST_PNG, "image/png", &plain())
            .expect("parent signs");
        let child = signer
            .sign(
                SELF_TEST_PNG,
                "image/png",
                &SignRequest {
                    ingredient: Some((&parent, "image/png")),
                    ai_edited: false,
                },
            )
            .expect("child signs");
        let summary = read_manifest(&child, "image/png").expect("the child carries a manifest");
        assert_eq!(summary.state, "valid");
        assert_eq!(summary.manifests, 2);
        assert_eq!(summary.ingredients, 1);
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
