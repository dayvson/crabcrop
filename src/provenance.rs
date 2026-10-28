//! C2PA provenance (issue #173): verify the Content
//! Credentials a source carries, and sign outputs when the
//! operator configured a signing identity.
//!
//! Design settled by the spike (2026-10-28):
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

use std::io::Cursor;

use c2pa::{Builder, Context, Reader, SigningAlg, ValidationState};

/// The IPTC digital source type recorded on the edit
/// action of an AI-edited output: the pixels were changed
/// with a trained-algorithm component, not synthesized
/// wholesale.
const AI_EDITED_SOURCE_TYPE: &str =
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia";

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

/// Which output formats v1 signs: the still formats the
/// spike proved end to end. GIF, JPEG XL, and video
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
/// authority URL. The signer object is rebuilt per sign
/// from the stored PEM bytes (parsing is sub-millisecond),
/// which keeps this struct plain data: shareable across
/// threads with no lock around the SDK's signer.
pub struct ProvenanceSigner {
    cert_pem: Vec<u8>,
    key_pem: Vec<u8>,
    alg: SigningAlg,
    tsa_url: Option<String>,
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
            .field("tsa_url", &self.tsa_url)
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
    /// starts answering.
    pub fn from_files(
        cert_path: &str,
        key_path: &str,
        tsa_url: Option<String>,
    ) -> Result<Self, String> {
        let cert_pem = std::fs::read(cert_path)
            .map_err(|e| format!("cannot read CRABCROP_C2PA_CERT file {cert_path}: {e}"))?;
        let key_pem = std::fs::read(key_path)
            .map_err(|e| format!("cannot read CRABCROP_C2PA_KEY file {key_path}: {e}"))?;
        Self::from_pem(cert_pem, key_pem, tsa_url)
    }

    /// Build and validate a signing identity from PEM
    /// bytes. Validation is a real round trip: sign the
    /// self-test image and require the result to read back
    /// valid. That one probe settles the algorithm (each
    /// candidate is tried until one produces a verifying
    /// signature), the certificate profile (the SDK refuses
    /// a bare self-signed certificate, an end-entity
    /// certificate without the required EKU, and any chain
    /// whose key does not match), and the timestamp
    /// authority when one is configured. A failure names
    /// the SDK's own error: boot fails fast, like the model
    /// preloads, instead of the first signed request
    /// discovering a broken identity.
    pub fn from_pem(
        cert_pem: Vec<u8>,
        key_pem: Vec<u8>,
        tsa_url: Option<String>,
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
            let signer = Self {
                cert_pem: cert_pem.clone(),
                key_pem: key_pem.clone(),
                alg,
                tsa_url: tsa_url.clone(),
                name: String::new(),
            };
            match signer.probe() {
                Ok(name) => {
                    return Ok(Self { name, ..signer });
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

    /// The boot probe: sign the self-test image with this
    /// identity and read the manifest back. Ok carries the
    /// signer name the readback reports.
    fn probe(&self) -> Result<String, String> {
        let signed = self.sign(
            SELF_TEST_PNG,
            "image/png",
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )?;
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
    /// are never modified.
    pub fn sign(
        &self,
        output: &[u8],
        mime: &str,
        request: &SignRequest<'_>,
    ) -> Result<Vec<u8>, String> {
        let signer = c2pa::create_signer::from_keys(
            &self.cert_pem,
            &self.key_pem,
            self.alg,
            self.tsa_url.clone(),
        )
        .map_err(|e| e.to_string())?;
        let definition = serde_json::json!({
            "title": "crabcrop output",
            "format": mime,
            "claim_generator_info": [{
                "name": "crabcrop",
                "version": env!("CARGO_PKG_VERSION"),
            }],
        })
        .to_string();
        let mut builder = Builder::from_context(Context::new())
            .with_definition(definition)
            .map_err(|e| e.to_string())?;
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
                .map_err(|e| e.to_string())?;
            builder
                .add_action(serde_json::json!({
                    "action": "c2pa.opened",
                    "parameters": { "ingredientIds": ["source"] },
                }))
                .map_err(|e| e.to_string())?;
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
                .map_err(|e| e.to_string())?;
        }
        if request.ai_edited {
            builder
                .add_action(serde_json::json!({
                    "action": "c2pa.edited",
                    "digitalSourceType": AI_EDITED_SOURCE_TYPE,
                }))
                .map_err(|e| e.to_string())?;
        }
        let mut source = Cursor::new(output);
        let mut dest = Cursor::new(Vec::new());
        builder
            .sign(&*signer, mime, &mut source, &mut dest)
            .map_err(|e| e.to_string())?;
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
// like a private operator CA.
#[cfg(test)]
pub(crate) const TEST_CHAIN: &str = "-----BEGIN CERTIFICATE-----\nMIIB4jCCAYmgAwIBAgIUcCXdPow++l0p0FSHJK0c0zO+/IEwCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMjgxODUxMTJaGA8yMTI2MTAwNDE4NTExMlowNzEdMBsGA1UE\nAwwUY3JhYmNyb3AgdGVzdCBzaWduZXIxFjAUBgNVBAoMDUNyYWJjcm9wIFRlc3Qw\nWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQeWY39h9h9eukk2qnrXHGqU7upywvS\n2ySnXDnczKViN62hVTtNVDyjwJQq7jaU/w2+/l5uztgBmPcNBv6vEHgYo3UwczAM\nBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQEAwIHgDATBgNVHSUEDDAKBggrBgEFBQcD\nBDAdBgNVHQ4EFgQU1rrLbUlCvE/gZ99fu6wAidIIFZIwHwYDVR0jBBgwFoAUqZMw\nTa6P7Eg+tNMjaGpBUfqat0kwCgYIKoZIzj0EAwIDRwAwRAIgdVWHqXBgtrLDtisG\nYaEtvsVjh28X/X3WC0Mbz+TGeZoCIEPwQFXP02YZNTBNI4qH1YIcw4MEuXMOzRpp\nbJxSXbTw\n-----END CERTIFICATE-----\n-----BEGIN CERTIFICATE-----\nMIIBvjCCAWOgAwIBAgIUc/mA9ToPXNgspCO2gfKnOpMb8SwwCgYIKoZIzj0EAwIw\nMzEZMBcGA1UEAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3Ag\nVGVzdDAgFw0yNjEwMjgxODUxMTJaGA8yMTI2MTAwNDE4NTExMlowMzEZMBcGA1UE\nAwwQY3JhYmNyb3AgdGVzdCBDQTEWMBQGA1UECgwNQ3JhYmNyb3AgVGVzdDBZMBMG\nByqGSM49AgEGCCqGSM49AwEHA0IABLEwHrQQMmP9/Q1BtF0JwPPjDhVH+vgbyjO2\nDOpW8ITW8JjgyxwK245BpXepHOFRX6Sf5av2uMQPdP8e+u0/DBqjUzBRMB0GA1Ud\nDgQWBBSpkzBNro/sSD600yNoakFR+pq3STAfBgNVHSMEGDAWgBSpkzBNro/sSD60\n0yNoakFR+pq3STAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMCA0kAMEYCIQDu\nxLkJML8djr0Vl20Vk3290vebT5umm6WgDgXzsUUQOQIhANsnF3kFIH5/M2hYt8uF\n87WSZPCiSXsWSaqK/HovPHh8\n-----END CERTIFICATE-----\n";
#[cfg(test)]
pub(crate) const TEST_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgDdKjjSDmVGisl6rO\nPWFhHIxOEZSN588utCGcu+P9cUGhRANCAAQeWY39h9h9eukk2qnrXHGqU7upywvS\n2ySnXDnczKViN62hVTtNVDyjwJQq7jaU/w2+/l5uztgBmPcNBv6vEHgY\n-----END PRIVATE KEY-----\n";

/// A test-identity signer for other modules' suites.
#[cfg(test)]
pub(crate) fn test_signer() -> std::sync::Arc<ProvenanceSigner> {
    std::sync::Arc::new(tests::test_signer())
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
