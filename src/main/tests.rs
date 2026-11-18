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
        crate::engine::AvifTiles::default(),
        crate::engine::DEFAULT_JXL_SPEED,
        DEFAULT_WIDTH_STEP,
    );
    assert_ne!(
        fingerprint, "7b993291c4880a4d",
        "the default fingerprint still matches 0.9.10"
    );
}

#[test]
fn default_configuration_keeps_its_fingerprint() {
    // The default configuration's fingerprint, with the
    // rendering version it was taken at. A change that moves
    // it starts a new cache epoch for every deployment on the
    // default configuration. Raising RENDERING_VERSION does
    // that on purpose: pin the new pair here (RELEASING.md,
    // step 4). A new setting should not: `CRABCROP_AVIF_TILES`
    // (issue #688) enters the fingerprint only when it names
    // another count than the default four, so this is the
    // value the default configuration had before it.
    let fingerprint = config_fingerprint(
        env_of(&[]),
        metadata_config(env_of(&[])).unwrap(),
        crate::quality::DEFAULT_SSIM_TARGET,
        crate::engine::DEFAULT_AVIF_SPEED,
        crate::engine::AvifTiles::default(),
        crate::engine::DEFAULT_JXL_SPEED,
        DEFAULT_WIDTH_STEP,
    );
    assert_eq!(
        (RENDERING_VERSION, fingerprint.as_str()),
        (8, "6e5d7c480309546a")
    );
}

#[test]
fn config_fingerprint_tracks_the_avif_tile_count() {
    // Issue #688: the tile count is part of what a URL renders
    // to, so each count is its own cache epoch, and replicas
    // that disagree on it keep separate entries instead of
    // serving each other's files.
    let speed = crate::engine::DEFAULT_JXL_SPEED;
    let mut seen = Vec::new();
    for count in [1, 2, 4, 8, 16] {
        let tiles = crate::engine::AvifTiles::new(count).unwrap();
        let fingerprint = config_fingerprint(env_of(&[]), false, 90.0, 6, tiles, speed, 100);
        assert!(!seen.contains(&fingerprint), "{count} tiles");
        seen.push(fingerprint);
    }
}

#[test]
fn config_fingerprint_tracks_output_shaping_settings() {
    // The fingerprint is a pure function of the
    // settings (issue #190, audit R2-08): stable for
    // identical configuration, different the moment an
    // output-shaping input changes, and blind to
    // settings that do not shape output.
    let speed = crate::engine::DEFAULT_JXL_SPEED;
    let tiles = crate::engine::AvifTiles::default();
    let base = config_fingerprint(env_of(&[]), false, 90.0, 6, tiles, speed, 100);
    assert_eq!(base.len(), 16);
    assert_eq!(
        base,
        config_fingerprint(env_of(&[]), false, 90.0, 6, tiles, speed, 100),
        "identical configuration, identical fingerprint"
    );
    assert_ne!(
        base,
        config_fingerprint(env_of(&[]), true, 90.0, 6, tiles, speed, 100),
        "the metadata policy shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(env_of(&[]), false, 80.0, 6, tiles, speed, 100),
        "the SSIM target shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(env_of(&[]), false, 90.0, 6, tiles, speed, 50),
        "the width step shapes output"
    );
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_REMOVEBG_MODEL", "/models/u2net.onnx")]),
            false,
            90.0,
            6,
            tiles,
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
            tiles,
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
            tiles,
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
            tiles,
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
            tiles,
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
        tiles,
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
        tiles,
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
    let tiles = crate::engine::AvifTiles::default();
    let base = config_fingerprint(env_of(&[]), false, 90.0, 6, tiles, speed, 100);
    let model = temp_file("fp-model", b"model weights, first build");
    let fp_model = config_fingerprint(
        env_of(&[("CRABCROP_REMOVEBG_MODEL", model.as_str())]),
        false,
        90.0,
        6,
        tiles,
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
        tiles,
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
            tiles,
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
        tiles,
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
            tiles,
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
        tiles,
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
            tiles,
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
    let tiles = crate::engine::AvifTiles::default();
    let base = config_fingerprint(env_of(&[]), false, 90.0, 6, tiles, speed, 100);
    assert_ne!(
        base,
        config_fingerprint(
            env_of(&[("CRABCROP_DETECT_THRESHOLD", "0.2")]),
            false,
            90.0,
            6,
            tiles,
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
            tiles,
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
            tiles,
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
            tiles,
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
    crate::test_script::write_script(&renderer_path, "#!/bin/sh\n");
    let renderer_str = renderer_path.to_string_lossy().into_owned();
    let err = expect_err(build_pdf_renderer(env_of(&[(
        "CRABCROP_PDF_RENDERER",
        renderer_str.as_str(),
    )])));
    assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
    assert!(err.contains("pdfinfo"), "{err}");
    // Issue #715: the pair in place runs but renders
    // nothing. Boot used to take it, and every `page:`
    // request then answered `502`; the boot check refuses it.
    let info_path = dir.join("pdfinfo");
    crate::test_script::write_script(&info_path, "#!/bin/sh\n");
    let err = expect_err(build_pdf_renderer(env_of(&[(
        "CRABCROP_PDF_RENDERER",
        renderer_str.as_str(),
    )])));
    assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
    assert!(
        err.contains("a one-page test document did not render"),
        "{err}"
    );
    // A pair that renders the test page constructs.
    let page = dir.join("page.png");
    image::RgbImage::new(16, 16).save(&page).unwrap();
    crate::test_script::write_script(
        &info_path,
        "#!/bin/sh\ncat > /dev/null\n\
         printf '%s\\n' 'Pages:           1' 'Encrypted:       no' 'Page    1 size:  8 x 8 pts'\n",
    );
    crate::test_script::write_script(
        &renderer_path,
        &format!("#!/bin/sh\ncat > /dev/null\ncat '{}'\n", page.display()),
    );
    let renderer = build_pdf_renderer(env_of(&[
        ("CRABCROP_PDF_RENDERER", renderer_str.as_str()),
        ("CRABCROP_PDF_TIMEOUT_SECS", "45"),
    ]))
    .unwrap();
    assert!(renderer.is_some());
}

/// Issue #715, with Poppler's own tools when they are on
/// PATH (a machine without them skips this). pdftoppm sits
/// next to pdfinfo and takes the renderer's arguments, so
/// boot took `CRABCROP_PDF_RENDERER` naming it. But it
/// writes its PNG to a file, `-.png` in its working
/// directory, and prints nothing, so every `page:` request
/// answered `502`. Boot now refuses it, and still takes
/// pdftocairo, the reference renderer.
#[cfg(unix)]
#[test]
fn pdf_renderer_construction_refuses_pdftoppm_and_takes_pdftocairo() {
    let Some(dir) = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path).find(|dir| {
            ["pdftocairo", "pdftoppm", "pdfinfo"]
                .iter()
                .all(|tool| dir.join(tool).is_file())
        })
    }) else {
        eprintln!("skipping: Poppler's pdftocairo, pdftoppm and pdfinfo are not on PATH");
        return;
    };
    let build = |tool: &str| {
        let path = dir.join(tool).to_string_lossy().into_owned();
        build_pdf_renderer(env_of(&[("CRABCROP_PDF_RENDERER", path.as_str())]))
    };
    assert!(build("pdftocairo").unwrap().is_some());
    // The tools run in the temporary directory, where
    // pdftoppm leaves its file.
    let left = std::env::temp_dir().join("-.png");
    let there = left.exists();
    let refused = build("pdftoppm");
    if !there {
        let _ = std::fs::remove_file(&left);
    }
    let err = expect_err(refused);
    assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
    assert!(err.contains("pdftoppm"), "{err}");
    assert!(err.contains("the PDF renderer produced no output"), "{err}");
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

/// A label set the CLIP loads in the tests below give back:
/// the labels it was handed, each scored alike, and a fixed
/// embedding.
#[derive(Debug)]
struct FakeLabelSet(Vec<String>);

impl crate::ai::TagProvider for FakeLabelSet {
    fn scores(
        &self,
        _img: &image::DynamicImage,
    ) -> Result<Vec<crate::ai::TagScore>, crate::ai::AiError> {
        Ok(self
            .0
            .iter()
            .map(|label| crate::ai::TagScore {
                label: label.clone(),
                score: 0.25,
            })
            .collect())
    }

    fn embedding(
        &self,
        _img: &image::DynamicImage,
    ) -> Result<Option<Vec<f32>>, crate::ai::AiError> {
        Ok(Some(vec![0.6, 0.8]))
    }
}

/// What a CLIP load gives back for `label_sets`, faked: a set
/// for each, in order.
fn fake_label_sets(label_sets: Vec<Vec<String>>) -> Vec<ClipLabelSet> {
    label_sets
        .into_iter()
        .map(|labels| {
            let count = labels.len();
            let set: Arc<dyn crate::ai::TagProvider> = Arc::new(FakeLabelSet(labels));
            (set, count)
        })
        .collect()
}

fn labels_of(scores: Vec<crate::ai::TagScore>) -> Vec<String> {
    scores.into_iter().map(|score| score.label).collect()
}

fn owned(labels: &[&str]) -> Vec<String> {
    labels.iter().map(|label| label.to_string()).collect()
}

#[test]
fn the_tags_and_the_clip_moderation_prompts_share_one_clip_load() {
    // Issue #736: with moderation on CLIP prompts, boot loaded
    // the CLIP models a second time for the prompts: two vision
    // sessions in memory, each with an embedding memo of its
    // own, so a meta request that asked for tags and moderation
    // embedded its image twice. Both label files now go to one
    // load, and the tags, the embedding and the review signals
    // are built over the sets it gives back.
    let tags = temp_file("clip-tags", b"beach\nforest\n");
    let prompts = temp_file("clip-prompts", b"# reviewed by hand\nweapon\nblood\n");
    let clip_env = [
        ("CRABCROP_CLIP_IMAGE_MODEL", "/models/clip-vision.onnx"),
        ("CRABCROP_CLIP_TEXT_MODEL", "/models/clip-text.onnx"),
        ("CRABCROP_CLIP_TOKENIZER", "/models/tokenizer.json"),
        ("CRABCROP_CLIP_LABELS", tags.as_str()),
    ];
    let with = |more: &[(&'static str, &str)]| {
        let pairs: Vec<(&str, &str)> = clip_env.iter().chain(more).copied().collect();
        ai_config(env_of(&pairs)).unwrap()
    };
    let config = with(&[("CRABCROP_MODERATION_LABELS", prompts.as_str())]);
    let mut loads: Vec<(String, Vec<Vec<String>>)> = Vec::new();
    let clip = build_clip_from(
        config.clip.as_ref(),
        config.moderation.as_ref(),
        |clip, label_sets| {
            loads.push((clip.image_model.clone(), label_sets.clone()));
            Ok(fake_label_sets(label_sets))
        },
    )
    .unwrap()
    .expect("the CLIP set is configured");
    assert_eq!(
        loads,
        [(
            "/models/clip-vision.onnx".to_string(),
            vec![owned(&["beach", "forest"]), owned(&["weapon", "blood"])]
        )]
    );
    let img = image::DynamicImage::new_rgb8(8, 8);
    assert_eq!(
        labels_of(clip.tagger.scores(&img).unwrap()),
        ["beach", "forest"]
    );
    let embedding = clip.embeddings.embed(&img).unwrap();
    assert_eq!(embedding.model, "clip-vision.onnx");
    assert_eq!(embedding.vector, [0.6, 0.8]);
    let report = clip
        .moderation
        .expect("moderation on CLIP prompts is built with the tags")
        .moderate(&img)
        .unwrap();
    assert_eq!(report.model, "clip-vision.onnx");
    assert!(report.scores.is_empty(), "{report:?}");
    assert_eq!(labels_of(report.signals), ["weapon", "blood"]);
    // A verdict model, or no moderation at all, leaves the
    // load the tags alone.
    for moderation in [
        &[("CRABCROP_MODERATION_MODEL", "/models/moderation.onnx")][..],
        &[],
    ] {
        let config = with(moderation);
        let mut loads = Vec::new();
        let clip = build_clip_from(
            config.clip.as_ref(),
            config.moderation.as_ref(),
            |_, label_sets| {
                loads.push(label_sets.clone());
                Ok(fake_label_sets(label_sets))
            },
        )
        .unwrap()
        .expect("the CLIP set is configured");
        assert_eq!(loads, [vec![owned(&["beach", "forest"])]], "{moderation:?}");
        assert!(clip.moderation.is_none(), "{moderation:?}");
    }
    // A prompts file that cannot be read still names its own
    // variable, and nothing loads.
    let missing = missing_path("clip-prompts");
    let config = with(&[("CRABCROP_MODERATION_LABELS", missing.as_str())]);
    let err = build_clip_from(
        config.clip.as_ref(),
        config.moderation.as_ref(),
        |_, _| -> Result<Vec<ClipLabelSet>, crate::ai::AiError> {
            panic!("the CLIP models loaded for an unreadable prompts file")
        },
    )
    .map(|_| ())
    .unwrap_err();
    assert!(
        matches!(
            err,
            ConfigError::Build {
                var: Some("CRABCROP_MODERATION_LABELS"),
                ..
            }
        ),
        "{err:?}"
    );
    assert!(
        err.to_string()
            .starts_with("cannot load the CLIP moderation fallback: cannot read the label file"),
        "{err}"
    );
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
        ("CRABCROP_AVIF_TILES", "16"),
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
    assert_eq!(state.avif_tiles.count(), 16);
    assert_eq!(state.config_fingerprint, config.config_fingerprint);
    assert_eq!(
        state.process_timeout,
        std::time::Duration::from_secs(config.process_timeout_secs)
    );
    assert!(state.store.is_none() && state.shared_store.is_none());
    assert!(state.providers.font.is_none() && state.fallback.is_none());
    // The request log is the queue and its writer thread
    // (issue #642), unless `CRABCROP_LOG_SYNC` asks for
    // each line where it is logged.
    assert_eq!(format!("{:?}", state.request_log), "RequestLog(queued)");
    let config = Config::from_env(env_of(&[
        ("CRABCROP_SECRET", "a production-grade secret"),
        ("CRABCROP_LOG_SYNC", "1"),
    ]))
    .unwrap();
    let state = build_state(&config, open_stores(&config).unwrap()).unwrap();
    assert_eq!(format!("{:?}", state.request_log), "RequestLog(inline)");
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

/// The minor page faults this thread has taken (Linux).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn minor_faults() -> u64 {
    let stat = std::fs::read_to_string("/proc/thread-self/stat").unwrap();
    // The fields after the command name, which is in
    // parentheses and may hold spaces: state, ppid, pgrp,
    // session, tty_nr, tpgid, flags, minflt.
    let fields = stat.rsplit_once(')').unwrap().1;
    fields.split_whitespace().nth(7).unwrap().parse().unwrap()
}

/// Run in a process of its own by
/// `raised_malloc_thresholds_keep_a_renders_blocks`: the
/// thresholds are the process's, and the other tests of
/// this binary move them. Prints the page faults of one
/// round of a render's blocks, once the first round is
/// over.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
#[ignore = "run in a process of its own by raised_malloc_thresholds_keep_a_renders_blocks"]
fn malloc_churn_helper() {
    if env::var("CRABCROP_TEST_RAISE_THRESHOLDS").as_deref() == Ok("1") {
        raise_malloc_thresholds();
    }
    // The frame-sized blocks of a resize from 2400x1600 to
    // 1600 wide: the decoded source, the resizer's
    // intermediate and the result, 24.3 MB together. Each
    // is filled, as a render fills it, and freed.
    let mut rounds = Vec::new();
    for _ in 0..8 {
        let before = minor_faults();
        let decoded = vec![1u8; 2400 * 1600 * 3];
        let intermediate = vec![2u8; 1600 * 1600 * 3];
        let result = vec![3u8; 1600 * 1067 * 3];
        std::hint::black_box((&decoded, &intermediate, &result));
        drop(decoded);
        drop(intermediate);
        drop(result);
        rounds.push(minor_faults() - before);
    }
    // The first round takes its pages for the first time
    // either way.
    let mut later = rounds[1..].to_vec();
    later.sort_unstable();
    println!("faults_per_round={}", later[later.len() / 2]);
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn raised_malloc_thresholds_keep_a_renders_blocks() {
    // Issue #690. A fresh process hands a render's blocks
    // back to the kernel when they total more than twice
    // the largest block it has freed, and takes them again
    // at the next render: one page fault for every 4 KiB,
    // about 5,900 for these 24.3 MB. With the thresholds
    // raised the blocks stay in the heap.
    let faults_per_round = |raise: &str| -> u64 {
        let out = std::process::Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::malloc_churn_helper",
                "--ignored",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("CRABCROP_TEST_RAISE_THRESHOLDS", raise)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{stdout}");
        // The count follows the test's name on its line.
        let count = stdout
            .split_once("faults_per_round=")
            .unwrap_or_else(|| panic!("the helper printed no count: {stdout}"))
            .1;
        count
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (fresh, raised) = (faults_per_round("0"), faults_per_round("1"));
    assert!(
        raised < 600,
        "{raised} page faults a round with the thresholds raised ({fresh} without)"
    );
    // Where the allocator does hand the blocks back, which
    // glibc does as long as its thresholds adjust
    // themselves, raising them is what stops it. An
    // allocator that keeps them anyway has nothing to show
    // here.
    if fresh >= 3000 {
        assert!(raised * 5 < fresh, "{raised} against {fresh}");
    }
}

#[test]
fn a_disk_budget_of_zero_opens_no_disk_tier_and_never_touches_the_directory() {
    // Issue #708: docs/configuration.md said a budget of 0
    // disables storage even with a directory set, but boot
    // still created and probed the directory, so one that was
    // not writable failed the boot. A budget of 0 is no disk
    // tier now: nothing creates, probes or reads the directory.
    // /proc is not writable even for a privileged process.
    let base = crate::test_support::scratch_dir("boot-disk-off");
    for dir in [
        "/proc/self".to_string(),
        base.join("never-made").display().to_string(),
    ] {
        let config = Config::from_env(env_of(&[
            ("CRABCROP_SECRET", "a production-grade secret"),
            ("CRABCROP_RESULT_STORAGE_DIR", dir.as_str()),
            ("CRABCROP_RESULT_STORAGE_MAX_MB", "0"),
        ]))
        .unwrap();
        let stores = open_stores(&config).unwrap();
        assert!(stores.disk.is_none(), "{dir}");
    }
    assert!(!base.join("never-made").exists());
    // With a budget, the same unwritable directory still stops
    // the boot, naming the variable.
    let config = Config::from_env(env_of(&[
        ("CRABCROP_SECRET", "a production-grade secret"),
        ("CRABCROP_RESULT_STORAGE_DIR", "/proc/self"),
    ]))
    .unwrap();
    let err = open_stores(&config).map(|_| ()).unwrap_err();
    assert!(
        matches!(
            err,
            ConfigError::Build {
                var: Some("CRABCROP_RESULT_STORAGE_DIR"),
                ..
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("is not writable"), "{err}");
}

#[test]
fn open_stores_starts_the_disk_tier_and_its_scan_indexes_what_a_previous_run_left() {
    // Issue #708: the store opens with its scan on a thread of
    // its own, and the boot waits for the scan up to the
    // store's deadline; a healthy disk is done by then.
    let dir = crate::test_support::scratch_dir("boot-disk-scan");
    let config = Config::from_env(env_of(&[
        ("CRABCROP_SECRET", "a production-grade secret"),
        ("CRABCROP_RESULT_STORAGE_DIR", &dir.display().to_string()),
    ]))
    .unwrap();
    let store = open_stores(&config).unwrap().disk.unwrap();
    let report = store.wait_scan(store.boot_deadline(), crate::storage::ScanPhase::ended);
    assert_eq!(report.phase, crate::storage::ScanPhase::Done);
    let image = crate::cache::CachedImage {
        bytes: bytes::Bytes::from_static(b"pixels"),
        format: crate::url::ImageFormat::Png,
        quality: None,
        groups: Vec::new(),
        etag: format!(
            "\"{}\"",
            hex::encode(&<sha2::Sha256 as sha2::Digest>::digest(b"pixels")[..16])
        ),
        written: None,
    };
    store.put("k", &image);
    drop(store);
    let store = open_stores(&config).unwrap().disk.unwrap();
    let report = store.wait_scan(store.boot_deadline(), crate::storage::ScanPhase::ended);
    assert_eq!(report.phase, crate::storage::ScanPhase::Done);
    assert_eq!((report.files, report.entries), (1, 1));
    assert!(store.get("k").is_some());
}

#[test]
fn the_disk_tier_boot_lines_say_what_the_scan_did() {
    // Issue #708: the boot logs when the scan starts, when it
    // is slow, and what it found once it ends, the files it
    // deleted included.
    use crate::storage::{ScanPhase, ScanReport};
    let report = |phase: ScanPhase, secs: u64| ScanReport {
        phase,
        elapsed: std::time::Duration::from_millis(secs * 1000 + 250),
        files: 12,
        entries: 9,
        bytes: 4096,
        deleted: 2,
        undeleted: 0,
    };
    assert_eq!(
        disk_tier_off("/var/cache/crabcrop"),
        "result storage: off, CRABCROP_RESULT_STORAGE_MAX_MB is 0, so /var/cache/crabcrop is not used"
    );
    assert_eq!(
        disk_scan_started("/c", 512, &ScanPhase::Scanning),
        "result storage at /c: budget 512 MB, scanning the entries a previous run left"
    );
    assert_eq!(
        disk_scan_started("/c", 512, &ScanPhase::Checking),
        "result storage at /c: budget 512 MB, waiting for the directory to answer"
    );
    assert_eq!(
        disk_scan_done("/c", 512, &report(ScanPhase::Done, 3)),
        "result storage at /c: 9 entries, 4096 bytes, budget 512 MB; the boot scan read 12 files in 3.2s, deleted 2 and skipped 1"
    );
    assert_eq!(
        disk_scan_done(
            "/c",
            512,
            &ScanReport {
                deleted: 1,
                undeleted: 1,
                ..report(ScanPhase::Done, 3)
            }
        ),
        "result storage at /c: 9 entries, 4096 bytes, budget 512 MB; the boot scan read 12 files in 3.2s, deleted 1, could not delete 1 and skipped 1"
    );
    assert_eq!(
        disk_scan_running("/c", &report(ScanPhase::Scanning, 35)),
        "result storage at /c: the boot scan is still running after 35s, 12 files read; until it ends, an entry it has not indexed is a miss"
    );
    assert_eq!(
        disk_scan_running("/c", &report(ScanPhase::Checking, 5)),
        "result storage at /c: the directory has not answered after 5s; until it does, the disk tier serves and stores nothing"
    );
    assert_eq!(
        disk_scan_failed(
            "/c",
            "cannot create the result storage directory: Permission denied"
        ),
        "result storage at /c: cannot create the result storage directory: Permission denied; the disk tier stays off until a restart"
    );
    assert_eq!(disk_scan_stopped(0).len(), 1);
    let lines = disk_scan_stopped(2);
    assert_eq!(lines.len(), 2);
    assert!(lines[1].starts_with("shutdown: 2 purges sent while the boot scan ran"));
}
