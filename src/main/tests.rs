use super::*;
use crate::config::tests::{env_of, expect_err, missing_path, temp_file};
use crate::config::{
    DEFAULT_WIDTH_STEP, MAX_DETECTORS_FILE_BYTES, ai_config, metadata_config, pdf_config,
    provenance_config, video_config,
};

// The construction tests drive each builder from an
// environment, as boot does: the configuration step
// first, then the build, so a refusal from either comes
// back the way boot reports it.

fn build_ai_providers(get: impl Fn(&str) -> Option<String>) -> Result<AiProviders, ConfigError> {
    super::build_ai_providers(&ai_config(get)?)
}

fn build_fallback(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<server::Fallback>, ConfigError> {
    super::build_fallback(
        get("CRABCROP_FALLBACK_IMAGE")
            .filter(|v| !v.is_empty())
            .as_deref(),
    )
}

fn build_video_encoder(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::video::VideoEncoder>>, ConfigError> {
    super::build_video_encoder(&video_config(get)?)
}

fn build_video_decoder(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::frames::VideoFrameExtractor>>, ConfigError> {
    super::build_video_decoder(&video_config(get)?)
}

fn build_pdf_renderer(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::pdf::PdfRenderer>>, ConfigError> {
    super::build_pdf_renderer(&pdf_config(get)?)
}

fn build_provenance_signer(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::provenance::ProvenanceSigner>>, ConfigError> {
    super::build_provenance_signer(provenance_config(get)?.as_ref())
}

#[test]
fn default_fingerprint_leaves_the_0_9_10_epoch() {
    // 0.9.10 printed `configuration fingerprint:
    // 7b993291c4880a4d` under the default
    // configuration. The release after it changes
    // rendered output for existing URLs, so the same
    // default configuration must print a different
    // fingerprint, or upgraded servers keep serving
    // 0.9.10 renders from the disk and shared tiers
    // (issue #412). The arguments are the boot path's
    // own defaults, not the fixed values the other
    // fingerprint tests use.
    let fingerprint = config_fingerprint(
        env_of(&[]),
        metadata_config(env_of(&[])).unwrap(),
        crate::quality::DEFAULT_SSIM_TARGET,
        crate::engine::DEFAULT_AVIF_SPEED,
        crate::engine::DEFAULT_JXL_SPEED,
        DEFAULT_WIDTH_STEP,
    );
    assert_ne!(
        fingerprint, "7b993291c4880a4d",
        "the default fingerprint still matches 0.9.10"
    );
}

#[test]
fn config_fingerprint_tracks_output_shaping_settings() {
    // The fingerprint is a pure function of the
    // settings (issue #190, audit R2-08): stable for
    // identical configuration, different the moment an
    // output-shaping input changes, and blind to
    // settings that do not shape output.
    let speed = crate::engine::DEFAULT_JXL_SPEED;
    let base = config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100);
    assert_eq!(base.len(), 16);
    assert_eq!(
        base,
        config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100),
        "identical configuration, identical fingerprint"
    );
    assert_ne!(
        base,
        config_fingerprint(env_of(&[]), true, 90.0, 6, speed, 100),
        "the metadata policy shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(env_of(&[]), false, 80.0, 6, speed, 100),
        "the SSIM target shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 50),
        "the width step shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_REMOVEBG_MODEL", "/models/u2net.onnx")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "a swapped model file shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_BRAND_COLOR", "f5f0e8")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "a brand color shapes studio output"
    );
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_C2PA_CERT", "/identity/cert.pem")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "a signing identity shapes signed output"
    );
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_C2PA_TSA_URL", "https://tsa.example.com")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "a timestamp authority shapes signatures"
    );
    assert_eq!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_PORT", "8080"), ("CRABCROP_CACHE_MAX_MB", "64")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "serving knobs do not shape output"
    );
    // The detectors file is identified by content:
    // editing a declaration changes the fingerprint,
    // and an unchanged file is stable.
    let file = temp_file("fp-detectors", br#"[{"name":"a"}]"#);
    let fp_file = config_fingerprint(
        env_of(&[("CRABCROP_DETECTORS_FILE", file.as_str())]),
        false,
        90.0,
        6,
        speed,
        100,
    );
    assert_ne!(base, fp_file);
    std::fs::write(&file, br#"[{"name":"b"}]"#).unwrap();
    let fp_edited = config_fingerprint(
        env_of(&[("CRABCROP_DETECTORS_FILE", file.as_str())]),
        false,
        90.0,
        6,
        speed,
        100,
    );
    assert_ne!(fp_file, fp_edited, "edited declarations, new epoch");
    let _ = std::fs::remove_file(&file);
}

#[test]
fn config_fingerprint_identifies_files_by_content() {
    // File-based inputs are identified by a digest of
    // their contents, not by their paths (issue #348):
    // replacing a file in place at the same path starts
    // a fresh epoch, and the same contents under a
    // different path do not.
    let speed = crate::engine::DEFAULT_JXL_SPEED;
    let base = config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100);
    let model = temp_file("fp-model", b"model weights, first build");
    let fp_model = config_fingerprint(
        env_of(&[("CRABCROP_REMOVEBG_MODEL", model.as_str())]),
        false,
        90.0,
        6,
        speed,
        100,
    );
    assert_ne!(base, fp_model);
    std::fs::write(&model, b"model weights, second build").unwrap();
    let fp_replaced = config_fingerprint(
        env_of(&[("CRABCROP_REMOVEBG_MODEL", model.as_str())]),
        false,
        90.0,
        6,
        speed,
        100,
    );
    assert_ne!(fp_model, fp_replaced, "replaced in place, new epoch");
    std::fs::write(&model, b"model weights, first build").unwrap();
    let moved = temp_file("fp-model-moved", b"model weights, first build");
    assert_eq!(
        fp_model,
        config_fingerprint(
            env_of(&[("CRABCROP_REMOVEBG_MODEL", moved.as_str())]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "same contents under a new path keep the epoch"
    );
    // The overlay font follows the same rule.
    let font = temp_file("fp-font", b"font bytes, first build");
    let fp_font = config_fingerprint(
        env_of(&[("CRABCROP_FONT_PATH", font.as_str())]),
        false,
        90.0,
        6,
        speed,
        100,
    );
    std::fs::write(&font, b"font bytes, second build").unwrap();
    assert_ne!(
        fp_font,
        config_fingerprint(
            env_of(&[("CRABCROP_FONT_PATH", font.as_str())]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "a replaced font starts a fresh epoch"
    );
    // A declared detector's model is identified by
    // content too: replacing it leaves the declarations
    // untouched and still starts a fresh epoch.
    let declared = temp_file("fp-declared-model", b"declared weights, first");
    let decls = temp_file(
        "fp-declared-file",
        format!(r#"[{{"name":"things","model":{declared:?},"kind":"boxes"}}]"#).as_bytes(),
    );
    let fp_declared = config_fingerprint(
        env_of(&[("CRABCROP_DETECTORS_FILE", decls.as_str())]),
        false,
        90.0,
        6,
        speed,
        100,
    );
    std::fs::write(&declared, b"declared weights, second").unwrap();
    assert_ne!(
        fp_declared,
        config_fingerprint(
            env_of(&[("CRABCROP_DETECTORS_FILE", decls.as_str())]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "a replaced declared model starts a fresh epoch"
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&moved);
    let _ = std::fs::remove_file(&font);
    let _ = std::fs::remove_file(&declared);
    let _ = std::fs::remove_file(&decls);
}

#[test]
fn config_fingerprint_tracks_ai_thresholds() {
    // The AI behavior thresholds shape output at their
    // effective values (issue #348): changing one
    // starts a fresh epoch, and setting one to its
    // default changes nothing.
    let speed = crate::engine::DEFAULT_JXL_SPEED;
    let base = config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100);
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_DETECT_THRESHOLD", "0.2")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "the detection threshold shapes redaction"
    );
    assert_eq!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_DETECT_THRESHOLD", "0.5")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "the default detection threshold is the unset fingerprint"
    );
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_CLIP_THRESHOLD", "0.2")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "the tagging threshold shapes annotations"
    );
    assert_eq!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_CLIP_THRESHOLD", "0.05")]),
            false,
            90.0,
            6,
            speed,
            100
        ),
        "the default tagging threshold is the unset fingerprint"
    );
}

#[test]
fn fallback_construction_reads_local_files_and_defers_urls() {
    assert!(build_fallback(env_of(&[])).unwrap().is_none());
    // An http(s) value is kept as a URL, fetched per event.
    let fallback = build_fallback(env_of(&[(
        "CRABCROP_FALLBACK_IMAGE",
        "https://example.com/fallback.png",
    )]))
    .unwrap()
    .unwrap();
    assert!(
        matches!(fallback, server::Fallback::Url(ref u) if u == "https://example.com/fallback.png")
    );
    // A missing local file is a startup error.
    let missing = missing_path("fallback");
    let err = expect_err(build_fallback(env_of(&[(
        "CRABCROP_FALLBACK_IMAGE",
        missing.as_str(),
    )])));
    assert!(err.contains("cannot read the fallback image"), "{err}");
    // A readable file is loaded once, bytes as-is: boot
    // does not decode, so undecodable bytes construct
    // fine here. Decode happens per fallback event, and
    // a failure there falls back to the original error.
    let path = temp_file("fallback", b"these are not image bytes");
    let fallback = build_fallback(env_of(&[("CRABCROP_FALLBACK_IMAGE", path.as_str())]))
        .unwrap()
        .unwrap();
    match fallback {
        server::Fallback::Bytes(bytes) => {
            assert_eq!(&bytes[..], b"these are not image bytes")
        }
        server::Fallback::Url(_) => panic!("a local path must load as bytes"),
    }
}

#[test]
fn video_construction_checks_the_bounds_even_without_an_encoder() {
    // Unconfigured and defaulted: off, no error.
    assert!(build_video_encoder(env_of(&[])).unwrap().is_none());
    // The bounds parse unconditionally: out-of-range or
    // unparseable values are boot errors even with no
    // encoder configured (issue #162).
    for (name, value) in [
        ("CRABCROP_VIDEO_TIMEOUT_SECS", "0"),
        ("CRABCROP_VIDEO_TIMEOUT_SECS", "601"),
        ("CRABCROP_VIDEO_TIMEOUT_SECS", "soon"),
        ("CRABCROP_VIDEO_MAX_MB", "0"),
        ("CRABCROP_VIDEO_MAX_MB", "1025"),
        ("CRABCROP_VIDEO_MAX_MB", "lots"),
    ] {
        let err = expect_err(build_video_encoder(env_of(&[(name, value)])));
        assert!(err.contains(name), "{name}={value}: {err}");
    }
    // The range edges are accepted.
    assert!(
        build_video_encoder(env_of(&[
            ("CRABCROP_VIDEO_TIMEOUT_SECS", "600"),
            ("CRABCROP_VIDEO_MAX_MB", "1"),
        ]))
        .unwrap()
        .is_none()
    );
}

#[cfg(unix)]
#[test]
fn video_construction_validates_the_encoder_program() {
    // A missing program is a boot error naming the variable.
    let missing = missing_path("encoder");
    let err = expect_err(build_video_encoder(env_of(&[(
        "CRABCROP_VIDEO_ENCODER",
        missing.as_str(),
    )])));
    assert!(err.contains("cannot use CRABCROP_VIDEO_ENCODER"), "{err}");
    assert!(err.contains("is not usable"), "{err}");
    // A directory is not a program.
    let dir = std::env::temp_dir().to_string_lossy().into_owned();
    let err = expect_err(build_video_encoder(env_of(&[(
        "CRABCROP_VIDEO_ENCODER",
        dir.as_str(),
    )])));
    assert!(err.contains("is not a file"), "{err}");
    // A non-executable file is refused...
    let plain = temp_file("encoder-plain", b"#!/bin/sh\n");
    let err = expect_err(build_video_encoder(env_of(&[(
        "CRABCROP_VIDEO_ENCODER",
        plain.as_str(),
    )])));
    assert!(err.contains("is not executable"), "{err}");
    // ...and the same file with the executable bit constructs.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o755)).unwrap();
    let encoder = build_video_encoder(env_of(&[
        ("CRABCROP_VIDEO_ENCODER", plain.as_str()),
        ("CRABCROP_VIDEO_TIMEOUT_SECS", "45"),
        ("CRABCROP_VIDEO_MAX_MB", "2"),
    ]))
    .unwrap();
    assert!(encoder.is_some());
}

#[test]
fn video_decoder_construction_follows_the_encoder_rules() {
    // Unset disables frame selection (issue #214).
    assert!(build_video_decoder(env_of(&[])).unwrap().is_none());
    // The shared timeout validates even when only the
    // decoder is configured.
    let err = expect_err(build_video_decoder(env_of(&[(
        "CRABCROP_VIDEO_TIMEOUT_SECS",
        "0",
    )])));
    assert!(err.contains("CRABCROP_VIDEO_TIMEOUT_SECS"), "{err}");
}

#[cfg(unix)]
#[test]
fn video_decoder_construction_validates_the_program() {
    // A missing program is a boot error naming the
    // variable, like the encoder's.
    let missing = missing_path("decoder");
    let err = expect_err(build_video_decoder(env_of(&[(
        "CRABCROP_VIDEO_DECODER",
        missing.as_str(),
    )])));
    assert!(err.contains("cannot use CRABCROP_VIDEO_DECODER"), "{err}");
    assert!(err.contains("does not exist"), "{err}");
    // A non-executable file is refused, and the same
    // file with the executable bit constructs.
    let plain = temp_file("decoder-plain", b"#!/bin/sh\n");
    let err = expect_err(build_video_decoder(env_of(&[(
        "CRABCROP_VIDEO_DECODER",
        plain.as_str(),
    )])));
    assert!(err.contains("is not executable"), "{err}");
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o755)).unwrap();
    let decoder = build_video_decoder(env_of(&[
        ("CRABCROP_VIDEO_DECODER", plain.as_str()),
        ("CRABCROP_VIDEO_TIMEOUT_SECS", "45"),
    ]))
    .unwrap();
    assert!(decoder.is_some());
}

#[test]
fn provenance_signer_construction_is_all_or_nothing() {
    // Unset: no signing (issue #173).
    assert!(build_provenance_signer(env_of(&[])).unwrap().is_none());
    // A timestamp authority alone is a configuration
    // error, not a silent no-op.
    let err = expect_err(build_provenance_signer(env_of(&[(
        "CRABCROP_C2PA_TSA_URL",
        "https://tsa.example.com",
    )])));
    assert!(err.contains("CRABCROP_C2PA_TSA_URL"), "{err}");
    // A partial pair names the missing half.
    let cert = temp_file("c2pa-cert", crate::provenance::TEST_CHAIN.as_bytes());
    let err = expect_err(build_provenance_signer(env_of(&[(
        "CRABCROP_C2PA_CERT",
        cert.as_str(),
    )])));
    assert!(err.contains("missing CRABCROP_C2PA_KEY"), "{err}");
    let key = temp_file("c2pa-key", crate::provenance::TEST_KEY.as_bytes());
    let err = expect_err(build_provenance_signer(env_of(&[(
        "CRABCROP_C2PA_KEY",
        key.as_str(),
    )])));
    assert!(err.contains("missing CRABCROP_C2PA_CERT"), "{err}");
    // The pair constructs, proven at boot by the
    // sign-and-verify round trip inside the build.
    let signer = build_provenance_signer(env_of(&[
        ("CRABCROP_C2PA_CERT", cert.as_str()),
        ("CRABCROP_C2PA_KEY", key.as_str()),
    ]))
    .unwrap()
    .expect("the pair configures a signer");
    assert_eq!(signer.name, "crabcrop test signer");
    // A set but unreadable certificate is fatal here.
    let missing = missing_path("c2pa-cert");
    let err = expect_err(build_provenance_signer(env_of(&[
        ("CRABCROP_C2PA_CERT", missing.as_str()),
        ("CRABCROP_C2PA_KEY", key.as_str()),
    ])));
    assert!(err.contains("CRABCROP_C2PA_CERT"), "{err}");
}

#[test]
fn provenance_boot_lines_name_the_algorithm_and_warn_only_for_rsa() {
    // Issue #600: the boot line names the algorithm the
    // identity settled on. An RSA key also gets a warning:
    // every signed output then runs an unblinded RSA
    // private-key operation in the rsa crate.
    use c2pa::SigningAlg;
    let line = |alg: &str| {
        format!("C2PA provenance signing: crabcrop test signer ({alg}, certificate /id/cert.pem)")
    };
    for (alg, name) in [
        (SigningAlg::Es256, "ES256"),
        (SigningAlg::Es384, "ES384"),
        (SigningAlg::Es512, "ES512"),
        (SigningAlg::Ed25519, "Ed25519"),
    ] {
        let (printed, warning) = provenance_boot_lines("crabcrop test signer", alg, "/id/cert.pem");
        assert_eq!(printed, line(name));
        assert_eq!(warning, None, "{name}");
    }
    for (alg, name) in [
        (SigningAlg::Ps256, "PS256"),
        (SigningAlg::Ps384, "PS384"),
        (SigningAlg::Ps512, "PS512"),
    ] {
        let (printed, warning) = provenance_boot_lines("crabcrop test signer", alg, "/id/cert.pem");
        assert_eq!(printed, line(name));
        let warning = warning.expect("an RSA key warns");
        for part in [name, "RSA", "RUSTSEC-2023-0071", "P-256", "Ed25519"] {
            assert!(warning.contains(part), "{part}: {warning}");
        }
    }
}

#[test]
fn pdf_renderer_construction_follows_the_decoder_rules() {
    // Unset disables page selection (issue #215).
    assert!(build_pdf_renderer(env_of(&[])).unwrap().is_none());
    // Its own timeout validates even with no renderer
    // configured.
    for value in ["0", "601", "soon"] {
        let err = expect_err(build_pdf_renderer(env_of(&[(
            "CRABCROP_PDF_TIMEOUT_SECS",
            value,
        )])));
        assert!(err.contains("CRABCROP_PDF_TIMEOUT_SECS"), "{value}: {err}");
    }
    assert!(
        build_pdf_renderer(env_of(&[("CRABCROP_PDF_TIMEOUT_SECS", "600")]))
            .unwrap()
            .is_none()
    );
}

#[cfg(unix)]
#[test]
fn pdf_renderer_construction_validates_both_programs() {
    use std::os::unix::fs::PermissionsExt;
    // A missing renderer is a boot error naming the
    // variable, like the decoder's.
    let missing = missing_path("renderer");
    let err = expect_err(build_pdf_renderer(env_of(&[(
        "CRABCROP_PDF_RENDERER",
        missing.as_str(),
    )])));
    assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
    assert!(err.contains("does not exist"), "{err}");
    // An executable renderer whose pdfinfo sibling is
    // missing is refused at boot too: the pair is the
    // contract.
    let dir = crate::test_support::scratch_dir("boot-pdf");
    let renderer_path = dir.join("pdftocairo");
    std::fs::write(&renderer_path, b"#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&renderer_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let renderer_str = renderer_path.to_string_lossy().into_owned();
    let err = expect_err(build_pdf_renderer(env_of(&[(
        "CRABCROP_PDF_RENDERER",
        renderer_str.as_str(),
    )])));
    assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
    assert!(err.contains("pdfinfo"), "{err}");
    // The pair in place constructs.
    let info_path = dir.join("pdfinfo");
    std::fs::write(&info_path, b"#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&info_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let renderer = build_pdf_renderer(env_of(&[
        ("CRABCROP_PDF_RENDERER", renderer_str.as_str()),
        ("CRABCROP_PDF_TIMEOUT_SECS", "45"),
    ]))
    .unwrap();
    assert!(renderer.is_some());
}

/// Issue #526: these construction tests once named
/// their files after the process ID alone, and a later
/// test process that drew a recycled ID found what an
/// earlier one had left: a `pdfinfo` next to the
/// renderer, scripts already marked executable. Their
/// "missing" and "not executable" refusals then
/// constructed. Plant those leftovers under this
/// process's ID and run the three tests over them, each
/// twice: under a name built from the ID alone, however
/// spelled, the second run would meet what the first one
/// left.
#[cfg(unix)]
#[test]
fn construction_tests_ignore_files_left_under_a_reused_pid() {
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt;
    let pid = std::process::id();
    let temp = std::env::temp_dir();
    let pdf_dir = temp.join(format!("crabcrop-boot-pdf-{pid}"));
    // Only what this test creates is removed afterwards:
    // a file already under one of these names is the same
    // kind of leftover, and not this test's to delete.
    let made_pdf_dir = std::fs::create_dir(&pdf_dir).is_ok();
    let mut planted = Vec::new();
    for path in [
        pdf_dir.join("pdfinfo"),
        temp.join(format!("crabcrop-boot-encoder-plain-{pid}.tmp")),
        temp.join(format!("crabcrop-boot-decoder-plain-{pid}.tmp")),
    ] {
        let Ok(mut file) = std::fs::File::create_new(&path) else {
            continue;
        };
        file.write_all(b"#!/bin/sh\n").unwrap();
        file.set_permissions(std::fs::Permissions::from_mode(0o755))
            .unwrap();
        planted.push(path);
    }
    let tests: [(&str, fn()); 3] = [
        (
            "pdf_renderer_construction_validates_both_programs",
            pdf_renderer_construction_validates_both_programs,
        ),
        (
            "video_construction_validates_the_encoder_program",
            video_construction_validates_the_encoder_program,
        ),
        (
            "video_decoder_construction_validates_the_program",
            video_decoder_construction_validates_the_program,
        ),
    ];
    let failed: Vec<&str> = tests
        .into_iter()
        .filter(|(_, test)| (0..2).any(|_| std::panic::catch_unwind(test).is_err()))
        .map(|(name, _)| name)
        .collect();
    for path in &planted {
        let _ = std::fs::remove_file(path);
    }
    if made_pdf_dir {
        let _ = std::fs::remove_dir(&pdf_dir);
    }
    assert!(
        failed.is_empty(),
        "failed over the planted files: {failed:?}"
    );
}

#[test]
fn ai_construction_builds_nothing_when_unconfigured() {
    let ai = build_ai_providers(env_of(&[])).unwrap();
    assert!(ai.removebg.is_none());
    assert!(ai.upscale.is_none());
    assert!(ai.face.is_none());
    assert!(ai.tagger.is_none());
    assert!(ai.moderation.is_none());
    assert!(ai.embeddings.is_none());
    assert!(ai.aesthetic.is_none());
    assert!(ai.ocr.is_none());
    assert!(ai.detect.is_none());
    assert!(ai.detectors.is_empty());
    assert!(ai.focal.is_none());
    // The default tag threshold still reaches the state.
    assert_eq!(ai.tag_threshold, 0.05);
}

#[test]
fn ai_construction_refuses_misconfiguration_before_any_load() {
    // The validation that runs before any model load is
    // gate-testable: no runtime is touched on these paths.
    let missing = missing_path("model");
    // A partial CLIP set refuses, naming what is missing.
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_CLIP_IMAGE_MODEL",
        missing.as_str(),
    )])));
    assert!(err.contains("partially configured"), "{err}");
    // An out-of-range CLIP threshold refuses too.
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_CLIP_IMAGE_MODEL", missing.as_str()),
        ("CRABCROP_CLIP_TEXT_MODEL", missing.as_str()),
        ("CRABCROP_CLIP_TOKENIZER", missing.as_str()),
        ("CRABCROP_CLIP_LABELS", missing.as_str()),
        ("CRABCROP_CLIP_THRESHOLD", "2.5"),
    ])));
    assert!(
        err.contains("CRABCROP_CLIP_THRESHOLD must be within 0..=1"),
        "{err}"
    );
    // The detector threshold is validated before the load.
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_DETECT_MODEL", missing.as_str()),
        ("CRABCROP_DETECT_THRESHOLD", "1.5"),
    ])));
    assert!(
        err.contains("CRABCROP_DETECT_THRESHOLD must be within 0..=1"),
        "{err}"
    );
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_DETECT_MODEL", missing.as_str()),
        ("CRABCROP_DETECT_THRESHOLD", "high"),
    ])));
    assert!(
        err.contains("CRABCROP_DETECT_THRESHOLD is not a number"),
        "{err}"
    );
    // Moderation labels without a model are zero-shot
    // prompts: without the CLIP set there is nothing to
    // score them with, and the refusal says so.
    let labels = missing_path("labels");
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_MODERATION_LABELS",
        labels.as_str(),
    )])));
    assert!(err.contains("need the CRABCROP_CLIP_* set"), "{err}");
    // With a model, an unreadable labels file refuses
    // before the model load is even attempted.
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_MODERATION_MODEL", missing.as_str()),
        ("CRABCROP_MODERATION_LABELS", labels.as_str()),
    ])));
    assert!(err.contains("cannot read the moderation labels"), "{err}");
    // An aesthetic head without the CLIP set refuses:
    // the head scores the tagger's embedding, and
    // there is no tagger to score with.
    let head = missing_path("head");
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_AESTHETIC_HEAD",
        head.as_str(),
    )])));
    assert!(
        err.contains("CRABCROP_AESTHETIC_HEAD is set but the CRABCROP_CLIP_* set"),
        "{err}"
    );
    // Half a segmentation pair refuses the same way
    // (issue #175): the provider is an encoder and a
    // prompt decoder, and one without the other
    // segments nothing.
    let model = missing_path("segment");
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_SEGMENT_ENCODER_MODEL",
        model.as_str(),
    )])));
    assert!(
        err.contains(
            "segmentation is partially configured: missing CRABCROP_SEGMENT_DECODER_MODEL"
        ),
        "{err}"
    );
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_SEGMENT_DECODER_MODEL",
        model.as_str(),
    )])));
    assert!(
        err.contains(
            "segmentation is partially configured: missing CRABCROP_SEGMENT_ENCODER_MODEL"
        ),
        "{err}"
    );
    // A partial OCR set refuses the same way (issue
    // #177): the pipeline is a detection model, a
    // recognition model, and its dictionary, and the
    // refusal names every missing piece.
    let model = missing_path("ocr");
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_OCR_DETECT_MODEL",
        model.as_str(),
    )])));
    assert!(
        err.contains(
            "OCR is partially configured: missing CRABCROP_OCR_RECOGNIZE_MODEL, CRABCROP_OCR_DICT"
        ),
        "{err}"
    );
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_OCR_DETECT_MODEL", model.as_str()),
        ("CRABCROP_OCR_RECOGNIZE_MODEL", model.as_str()),
    ])));
    assert!(
        err.contains("OCR is partially configured: missing CRABCROP_OCR_DICT"),
        "{err}"
    );
    // The language tag alone configures nothing.
    let ai = build_ai_providers(env_of(&[("CRABCROP_OCR_LANGUAGE", "en")])).unwrap();
    assert!(ai.ocr.is_none());
}

/// Configured-but-unloadable models refuse construction
/// with the boot wording. Ignored, gated exactly like the
/// src/ai.rs load tests (issue #179, item 2): a load only
/// reaches the model file once the runtime initializes
/// from an explicit dylib; with no dylib, ort's default
/// init panics inside the crate instead of erroring, so
/// the file path cannot be exercised in the plain gate.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored ai_construction_refuses_configured
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn ai_construction_refuses_configured_models_that_do_not_load() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let missing = missing_path("model");
    for (name, wording) in [
        (
            "CRABCROP_REMOVEBG_MODEL",
            "cannot load the background-removal model",
        ),
        ("CRABCROP_UPSCALE_MODEL", "cannot load the upscale model"),
        ("CRABCROP_INPAINT_MODEL", "cannot load the inpainting model"),
        (
            "CRABCROP_FACE_MODEL",
            "cannot load the face detection model",
        ),
        ("CRABCROP_DETECT_MODEL", "cannot load the detection model"),
    ] {
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_ORT_DYLIB", dylib.as_str()),
            (name, missing.as_str()),
        ])));
        assert!(err.contains(wording), "{name}: {err}");
    }
    // A full CLIP set whose files do not exist refuses too.
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_ORT_DYLIB", dylib.as_str()),
        ("CRABCROP_CLIP_IMAGE_MODEL", missing.as_str()),
        ("CRABCROP_CLIP_TEXT_MODEL", missing.as_str()),
        ("CRABCROP_CLIP_TOKENIZER", missing.as_str()),
        ("CRABCROP_CLIP_LABELS", missing.as_str()),
    ])));
    assert!(err.contains("cannot load the CLIP tagging models"), "{err}");
    // A full segmentation pair whose files do not
    // exist refuses too (issue #175).
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_ORT_DYLIB", dylib.as_str()),
        ("CRABCROP_SEGMENT_ENCODER_MODEL", missing.as_str()),
        ("CRABCROP_SEGMENT_DECODER_MODEL", missing.as_str()),
    ])));
    assert!(err.contains("cannot load the segment model pair"), "{err}");
    // A full OCR set whose files do not exist refuses
    // too (issue #177).
    let err = expect_err(build_ai_providers(env_of(&[
        ("CRABCROP_ORT_DYLIB", dylib.as_str()),
        ("CRABCROP_OCR_DETECT_MODEL", missing.as_str()),
        ("CRABCROP_OCR_RECOGNIZE_MODEL", missing.as_str()),
        ("CRABCROP_OCR_DICT", missing.as_str()),
    ])));
    assert!(err.contains("cannot load the OCR models"), "{err}");
}

/// Full AI construction with real models: every
/// configured slot of the provider set is populated.
/// Ignored like the src/ai.rs live tests: it needs the
/// runtime and the model files, named by the same
/// CRABCROP_TEST_* variables those tests use:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// CRABCROP_TEST_UPSCALE_MODEL=... CRABCROP_TEST_FACE_MODEL=...
/// CRABCROP_TEST_DETECT_MODEL=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
/// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
/// CRABCROP_TEST_CLIP_LABELS=...
/// cargo test -- --ignored ai_construction_with_real_models
#[test]
#[ignore = "needs the ONNX runtime and model files (env-configured)"]
fn ai_construction_with_real_models_builds_every_provider() {
    let var = |name: &str| std::env::var(name).unwrap();
    let mut map: std::collections::HashMap<String, String> = [
        ("CRABCROP_ORT_DYLIB", "CRABCROP_TEST_ORT_DYLIB"),
        ("CRABCROP_REMOVEBG_MODEL", "CRABCROP_TEST_REMOVEBG_MODEL"),
        ("CRABCROP_UPSCALE_MODEL", "CRABCROP_TEST_UPSCALE_MODEL"),
        ("CRABCROP_FACE_MODEL", "CRABCROP_TEST_FACE_MODEL"),
        ("CRABCROP_DETECT_MODEL", "CRABCROP_TEST_DETECT_MODEL"),
        (
            "CRABCROP_CLIP_IMAGE_MODEL",
            "CRABCROP_TEST_CLIP_IMAGE_MODEL",
        ),
        ("CRABCROP_CLIP_TEXT_MODEL", "CRABCROP_TEST_CLIP_TEXT_MODEL"),
        ("CRABCROP_CLIP_TOKENIZER", "CRABCROP_TEST_CLIP_TOKENIZER"),
        ("CRABCROP_CLIP_LABELS", "CRABCROP_TEST_CLIP_LABELS"),
    ]
    .iter()
    .map(|(config, test)| (config.to_string(), var(test)))
    .collect();
    if let Ok(labels) = std::env::var("CRABCROP_TEST_DETECT_LABELS") {
        map.insert("CRABCROP_DETECT_LABELS".to_string(), labels);
    }
    // Moderation joins when its test model is provided
    // (CRABCROP_TEST_MODERATION_MODEL), like the
    // detector labels above.
    let want_moderation = if let Ok(model) = std::env::var("CRABCROP_TEST_MODERATION_MODEL") {
        map.insert("CRABCROP_MODERATION_MODEL".to_string(), model);
        true
    } else {
        false
    };
    // The aesthetic head joins the same way
    // (CRABCROP_TEST_AESTHETIC_HEAD): a pure-Rust
    // load, on top of the CLIP pair above.
    let want_aesthetic = if let Ok(head) = std::env::var("CRABCROP_TEST_AESTHETIC_HEAD") {
        map.insert("CRABCROP_AESTHETIC_HEAD".to_string(), head);
        true
    } else {
        false
    };
    // Inpainting joins the same way
    // (CRABCROP_TEST_INPAINT_MODEL).
    let want_inpaint = if let Ok(model) = std::env::var("CRABCROP_TEST_INPAINT_MODEL") {
        map.insert("CRABCROP_INPAINT_MODEL".to_string(), model);
        true
    } else {
        false
    };
    // Segmentation joins the same way, as a pair
    // (CRABCROP_TEST_SEGMENT_ENCODER_MODEL and
    // CRABCROP_TEST_SEGMENT_DECODER_MODEL).
    let want_segment = match (
        std::env::var("CRABCROP_TEST_SEGMENT_ENCODER_MODEL"),
        std::env::var("CRABCROP_TEST_SEGMENT_DECODER_MODEL"),
    ) {
        (Ok(encoder), Ok(decoder)) => {
            map.insert("CRABCROP_SEGMENT_ENCODER_MODEL".to_string(), encoder);
            map.insert("CRABCROP_SEGMENT_DECODER_MODEL".to_string(), decoder);
            true
        }
        _ => false,
    };
    // OCR joins the same way, as a set of three paths
    // (CRABCROP_TEST_OCR_DETECT_MODEL,
    // CRABCROP_TEST_OCR_RECOGNIZE_MODEL,
    // CRABCROP_TEST_OCR_DICT).
    let want_ocr = match (
        std::env::var("CRABCROP_TEST_OCR_DETECT_MODEL"),
        std::env::var("CRABCROP_TEST_OCR_RECOGNIZE_MODEL"),
        std::env::var("CRABCROP_TEST_OCR_DICT"),
    ) {
        (Ok(detect), Ok(recognize), Ok(dict)) => {
            map.insert("CRABCROP_OCR_DETECT_MODEL".to_string(), detect);
            map.insert("CRABCROP_OCR_RECOGNIZE_MODEL".to_string(), recognize);
            map.insert("CRABCROP_OCR_DICT".to_string(), dict);
            true
        }
        _ => false,
    };
    let ai = build_ai_providers(|name: &str| map.get(name).cloned()).unwrap();
    assert!(ai.removebg.is_some());
    assert!(ai.upscale.is_some());
    assert_eq!(ai.inpaint.is_some(), want_inpaint);
    assert_eq!(ai.segment.is_some(), want_segment);
    assert_eq!(ai.ocr.is_some(), want_ocr);
    assert!(ai.face.is_some());
    assert!(ai.tagger.is_some());
    // The embedding provider rides the tagger.
    assert!(ai.embeddings.is_some());
    assert!(ai.detect.is_some());
    assert_eq!(ai.moderation.is_some(), want_moderation);
    assert_eq!(ai.aesthetic.is_some(), want_aesthetic);
}

#[test]
fn declared_detectors_refuse_an_oversized_file() {
    // A declarations file past the byte cap refuses at
    // the read, before any parsing or load (audit R2-27).
    let oversized = temp_file(
        "detectors-oversized",
        &vec![b' '; MAX_DETECTORS_FILE_BYTES + 1],
    );
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_DETECTORS_FILE",
        oversized.as_str(),
    )])));
    assert!(err.contains("byte cap"), "{err}");
}

#[test]
fn declared_detectors_refuse_files_that_fail_before_any_load() {
    // These refusals precede model contact, so they run
    // in the plain gate through full construction.
    let missing = missing_path("detectors");
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_DETECTORS_FILE",
        missing.as_str(),
    )])));
    assert!(err.contains("cannot read CRABCROP_DETECTORS_FILE"), "{err}");
    let bad_kind = temp_file(
        "detectors-kind",
        br#"[{"name": "crates", "model": "/m.onnx", "kind": "heatmap"}]"#,
    );
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_DETECTORS_FILE",
        bad_kind.as_str(),
    )])));
    assert!(err.contains("unknown kind \"heatmap\""), "{err}");
    assert!(err.contains("\"crates\""), "{err}");
    let dup = temp_file(
        "detectors-dup",
        br#"[{"name": "a", "model": "/m.onnx", "kind": "boxes"},
                 {"name": "a", "model": "/n.onnx", "kind": "focal"}]"#,
    );
    let err = expect_err(build_ai_providers(env_of(&[(
        "CRABCROP_DETECTORS_FILE",
        dup.as_str(),
    )])));
    assert!(err.contains("duplicate detector name: a"), "{err}");
}

/// Declared detectors with real models (issue #166):
/// a boxes declaration with inline classes whose count
/// matches the export, and two focal declarations (the
/// second stands by) build into the provider set.
/// Ignored like the other construction tests:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
/// CRABCROP_TEST_FOCAL_MODEL=...
/// cargo test -- --ignored declared_detectors_build_from
#[test]
#[ignore = "needs the ONNX runtime and model files (env-configured)"]
fn declared_detectors_build_from_a_file_with_real_models() {
    let Some(dylib) = crate::ai::env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(yolo) = crate::ai::env_or_skip("CRABCROP_TEST_DETECT_MODEL") else {
        return;
    };
    let Some(focal_model) = crate::ai::env_or_skip("CRABCROP_TEST_FOCAL_MODEL") else {
        return;
    };
    // The declaration's vocabulary must match the
    // export's class count (the load probe enforces
    // it): 80 generated names.
    let names: Vec<String> = (0..80).map(|i| format!("\"class{i}\"")).collect();
    // The same vocabulary as a labels file, its first
    // name renamed: labels in path form win over the
    // model's metadata exactly like the inline form.
    let labels_text: String = (0..80)
        .map(|i| {
            if i == 0 {
                "human\n".to_string()
            } else {
                format!("class{i}\n")
            }
        })
        .collect();
    let labels_file = temp_file("detectors-labels", labels_text.as_bytes());
    let text = format!(
        r#"[
              {{"name": "things", "model": "{yolo}", "kind": "boxes",
                "labels": [{}], "threshold": 0.4}},
              {{"name": "things-file", "model": "{yolo}", "kind": "boxes",
                "labels": "{labels_file}"}},
              {{"name": "saliency", "model": "{focal_model}", "kind": "focal"}},
              {{"name": "saliency-2", "model": "{focal_model}", "kind": "focal"}}
            ]"#,
        names.join(",")
    );
    let file = temp_file("detectors-build", text.as_bytes());
    let ai = build_ai_providers(env_of(&[
        ("CRABCROP_ORT_DYLIB", dylib.as_str()),
        ("CRABCROP_DETECTORS_FILE", file.as_str()),
    ]))
    .unwrap();
    assert_eq!(ai.detectors.len(), 2);
    assert_eq!(ai.detectors[0].name, "things");
    assert_eq!(
        crate::ai::DetectionProvider::describe(&*ai.detectors[0].provider).len(),
        80
    );
    assert_eq!(ai.detectors[1].name, "things-file");
    assert_eq!(
        crate::ai::DetectionProvider::describe(&*ai.detectors[1].provider)[0],
        "human"
    );
    assert!(ai.focal.is_some());
}

/// Each load-level refusal of a declared detector
/// surfaces at construction, naming the declaration
/// (issue #166). Ignored like the other load tests:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
/// CRABCROP_TEST_FACE_MODEL=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored declared_detectors_refuse_unloadable
#[test]
#[ignore = "needs the ONNX runtime and model files (env-configured)"]
fn declared_detectors_refuse_unloadable_models_at_boot() {
    let var = |name: &str| std::env::var(name).unwrap();
    let dylib = var("CRABCROP_TEST_ORT_DYLIB");
    let missing = missing_path("declared-model");
    let build = |text: String| {
        let file = temp_file("detectors-refuse", text.as_bytes());
        expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_ORT_DYLIB", dylib.as_str()),
            ("CRABCROP_DETECTORS_FILE", file.as_str()),
        ])))
    };
    // A boxes declaration whose model file is missing.
    let err = build(format!(
        r#"[{{"name": "crates", "model": "{missing}", "kind": "boxes", "labels": ["crate"]}}]"#
    ));
    assert!(
        err.contains("cannot load declared detector \"crates\" (boxes)"),
        "{err}"
    );
    // A boxes declaration with no vocabulary anywhere:
    // the model carries no names metadata and the
    // declaration gives no labels.
    let removebg = var("CRABCROP_TEST_REMOVEBG_MODEL");
    let err = build(format!(
        r#"[{{"name": "crates", "model": "{removebg}", "kind": "boxes"}}]"#
    ));
    assert!(
        err.contains("cannot load declared detector \"crates\" (boxes)"),
        "{err}"
    );
    assert!(err.contains("carries no class names metadata"), "{err}");
    // A focal declaration on a multi-output model fails
    // the heatmap probe.
    let yunet = var("CRABCROP_TEST_FACE_MODEL");
    let err = build(format!(
        r#"[{{"name": "saliency", "model": "{yunet}", "kind": "focal"}}]"#
    ));
    assert!(
        err.contains("cannot load declared detector \"saliency\" (focal)"),
        "{err}"
    );
    assert!(err.contains("does not produce a heatmap output"), "{err}");
}

#[test]
fn build_state_builds_the_configured_state_and_sizes_the_gates() {
    let config = Config::from_env(env_of(&[
        ("CRABCROP_SECRET", "a production-grade secret"),
        ("CRABCROP_METRICS_TOKEN", "metrics-token"),
        ("CRABCROP_SLOW_LANE_PERMITS", "3"),
        ("CRABCROP_META_MODE", "1"),
        ("CRABCROP_CLIENT_HINTS", "0"),
        ("CRABCROP_EXPLAIN", "0"),
    ]))
    .unwrap();
    let state = build_state(&config, open_stores(&config).unwrap()).unwrap();
    // The gate sizes are the documented constants.
    assert_eq!(state.ai_gate.available_permits(), AI_GATE_PERMITS);
    assert_eq!(state.anim_gate.available_permits(), ANIM_GATE_PERMITS);
    assert_eq!(state.probe_gate.available_permits(), PROBE_GATE_PERMITS);
    assert_eq!(
        state.video_gate.available_permits(),
        crate::video::GATE_PERMITS
    );
    assert_eq!(state.process_gate.capacity(), config.process_permits);
    assert_eq!(state.slow_gate.capacity(), 3);
    // The settings reach the state as validated.
    assert_eq!(state.metrics_token.as_deref(), Some("metrics-token"));
    assert!(state.meta_enabled);
    assert!(!state.client_hints);
    assert!(!state.explain_enabled);
    assert_eq!(state.config_fingerprint, config.config_fingerprint);
    assert_eq!(
        state.process_timeout,
        std::time::Duration::from_secs(config.process_timeout_secs)
    );
    assert!(state.store.is_none() && state.shared_store.is_none());
    assert!(state.providers.font.is_none() && state.fallback.is_none());
}

#[test]
fn build_state_names_a_component_that_does_not_build() {
    let font = temp_file("state-font", b"not a font");
    let config = Config::from_env(env_of(&[
        ("CRABCROP_SECRET", "a production-grade secret"),
        ("CRABCROP_FONT_PATH", font.as_str()),
    ]))
    .unwrap();
    let err = build_state(&config, open_stores(&config).unwrap())
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(
            err,
            ConfigError::Build {
                var: Some("CRABCROP_FONT_PATH"),
                ..
            }
        ),
        "{err:?}"
    );
    assert!(
        err.to_string()
            .starts_with("cannot load the overlay font: "),
        "{err}"
    );
}
