//! Unit tests for `config.rs`: each setting's parsing and
//! validation, driven through an injected lookup, and the
//! helpers the construction tests in `main.rs` share.

use super::*;

#[test]
fn brand_color_config_parses_hex_and_rejects_the_rest() {
    let none = |_: &str| None;
    assert_eq!(
        brand_color_config(none).map_err(|e| e.to_string()),
        Ok(None)
    );
    let plain = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(|| "f5f0e8".to_string());
    assert_eq!(
        brand_color_config(plain).map_err(|e| e.to_string()),
        Ok(Some([0xf5, 0xf0, 0xe8]))
    );
    // A leading # and uppercase hex are the same color.
    let hashed = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(|| "#F5F0E8".to_string());
    assert_eq!(
        brand_color_config(hashed).map_err(|e| e.to_string()),
        Ok(Some([0xf5, 0xf0, 0xe8]))
    );
    // Set but malformed is a startup error naming the
    // variable, never a silent default.
    for bad in ["f5f0e", "f5f0e81", "zzzzzz", "white"] {
        let get = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(|| bad.to_string());
        let err = brand_color_config(get).unwrap_err().to_string();
        assert!(err.contains("CRABCROP_BRAND_COLOR"), "{err}");
    }
    // An empty value is unset, like the other settings.
    let empty = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(String::new);
    assert_eq!(
        brand_color_config(empty).map_err(|e| e.to_string()),
        Ok(None)
    );
}

#[test]
fn metadata_config_defaults_to_strip() {
    let none = |_: &str| None;
    assert_eq!(metadata_config(none).map_err(|e| e.to_string()), Ok(false));
    let preserve = |name: &str| (name == "CRABCROP_METADATA").then(|| "preserve".to_string());
    assert_eq!(
        metadata_config(preserve).map_err(|e| e.to_string()),
        Ok(true)
    );
    let strip = |name: &str| (name == "CRABCROP_METADATA").then(|| "STRIP".to_string());
    assert_eq!(metadata_config(strip).map_err(|e| e.to_string()), Ok(false));
    let bogus = |name: &str| (name == "CRABCROP_METADATA").then(|| "keep".to_string());
    assert!(metadata_config(bogus).is_err());
}

#[test]
fn meta_mode_defaults_to_off_and_turns_on_by_the_boolean_spellings() {
    // Issue #453: unset is off.
    assert_eq!(
        meta_mode_config(env_of(&[])).map_err(|e| e.to_string()),
        Ok(false)
    );
    let set =
        |v: &str| meta_mode_config(env_of(&[("CRABCROP_META_MODE", v)])).map_err(|e| e.to_string());
    for v in ["1", "true", "YES", "on"] {
        assert_eq!(set(v), Ok(true), "{v}");
    }
    for v in ["0", "false", "no", "OFF"] {
        assert_eq!(set(v), Ok(false), "{v}");
    }
    // A typo, an empty value, or a stray space is a boot
    // error naming the variable, never a silent default.
    for bad in ["", " 1", "enabled", "2", "on\n"] {
        let err = expect_err(set(bad));
        assert!(err.contains("CRABCROP_META_MODE"), "{bad:?}: {err}");
    }
}

#[test]
fn clip_config_needs_all_four_paths_or_none() {
    let get = |pairs: &[(&str, &str)]| {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    };
    // None configured: off.
    assert!(clip_config(get(&[])).unwrap().is_none());
    // Partial: an error naming what is missing.
    let err = clip_config(get(&[("CRABCROP_CLIP_IMAGE_MODEL", "a.onnx")]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("CRABCROP_CLIP_TEXT_MODEL"), "{err}");
    assert!(err.contains("CRABCROP_CLIP_LABELS"), "{err}");
    // Full set: defaults and explicit threshold.
    let full = clip_config(get(&[
        ("CRABCROP_CLIP_IMAGE_MODEL", "a.onnx"),
        ("CRABCROP_CLIP_TEXT_MODEL", "b.onnx"),
        ("CRABCROP_CLIP_TOKENIZER", "t.json"),
        ("CRABCROP_CLIP_LABELS", "labels.txt"),
    ]))
    .unwrap()
    .unwrap();
    assert_eq!(full.threshold, 0.05);
    let bad = clip_config(get(&[
        ("CRABCROP_CLIP_IMAGE_MODEL", "a.onnx"),
        ("CRABCROP_CLIP_TEXT_MODEL", "b.onnx"),
        ("CRABCROP_CLIP_TOKENIZER", "t.json"),
        ("CRABCROP_CLIP_LABELS", "labels.txt"),
        ("CRABCROP_CLIP_THRESHOLD", "2.5"),
    ]));
    assert!(bad.is_err());
}

#[test]
fn bucket_policy_fails_closed_with_credentials() {
    assert!(check_bucket_policy(false, &[]).is_ok());
    assert!(check_bucket_policy(false, &["photos".to_string()]).is_ok());
    assert!(check_bucket_policy(true, &["photos".to_string()]).is_ok());
    assert!(check_bucket_policy(true, &["*".to_string()]).is_ok());
    let err = check_bucket_policy(true, &[]).unwrap_err().to_string();
    assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
}

#[test]
fn source_policy_blocks_private_addresses_by_default() {
    let get = |pairs: &[(&str, &str)]| {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    };
    // No configuration: no allowlist, blocking on. Boot
    // needs no posture setting.
    let policy = build_source_policy(get(&[])).unwrap();
    assert!(policy.allowed_suffixes.is_empty());
    assert!(policy.block_private);
    // The explicit switch turns blocking off.
    let policy = build_source_policy(get(&[("CRABCROP_BLOCK_PRIVATE_SOURCES", "0")])).unwrap();
    assert!(!policy.block_private);
    // The any-source opt-out turns blocking off even
    // with the switch on.
    let policy = build_source_policy(get(&[
        ("CRABCROP_BLOCK_PRIVATE_SOURCES", "1"),
        ("CRABCROP_ALLOW_ANY_SOURCE", "1"),
    ]))
    .unwrap();
    assert!(!policy.block_private);
    // The allowlist passes through normalized, and blank
    // entries normalize away.
    let policy = build_source_policy(get(&[(
        "CRABCROP_ALLOWED_SOURCES",
        " Example.com , ,127.0.0.1 ",
    )]))
    .unwrap();
    assert_eq!(policy.allowed_suffixes, ["example.com", "127.0.0.1"]);
    assert!(policy.block_private);
    // A bad boolean is a startup error naming the
    // variable, never a silent default.
    let err = build_source_policy(get(&[("CRABCROP_BLOCK_PRIVATE_SOURCES", "ture")]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("CRABCROP_BLOCK_PRIVATE_SOURCES"), "{err}");
}

#[test]
fn a_proxy_that_pinned_fetches_bypass_is_warned_about() {
    let proxied = |name: &'static str| {
        move |n: &str| (n == name).then(|| "http://proxy.internal:3128".to_string())
    };
    // The default policy pins every source hostname to
    // its validated address (issue #471): a proxy set
    // alongside it is named in a warning.
    let default_policy = build_source_policy(env_of(&[])).unwrap();
    let warning = proxy_bypass_warning(
        env_of(&[("HTTPS_PROXY", "http://user:hunter2@proxy.internal:3128")]),
        &default_policy,
    )
    .expect("a warning under the default policy");
    assert!(warning.contains("(HTTPS_PROXY)"), "{warning}");
    assert!(warning.contains("CRABCROP_ALLOWED_SOURCES"), "{warning}");
    // The value is never echoed: a proxy URL can carry
    // credentials.
    assert!(!warning.contains("hunter2"), "{warning}");
    assert!(!warning.contains("proxy.internal"), "{warning}");
    // Every spelling the HTTP client reads counts.
    for name in [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ] {
        let warning = proxy_bypass_warning(proxied(name), &default_policy)
            .unwrap_or_else(|| panic!("no warning for {name}"));
        assert!(warning.contains(&format!("({name})")), "{warning}");
    }
    let warning = proxy_bypass_warning(
        env_of(&[("HTTPS_PROXY", "http://p:1"), ("all_proxy", "http://p:1")]),
        &default_policy,
    )
    .unwrap();
    assert!(warning.contains("(HTTPS_PROXY, all_proxy)"), "{warning}");
    // No proxy is nothing to warn about, and neither is
    // an empty one: the client reads it as none, and an
    // empty uppercase spelling hides the lowercase one.
    for pairs in [
        &[][..],
        &[("HTTPS_PROXY", "")],
        &[("HTTPS_PROXY", ""), ("https_proxy", "http://p:1")],
        &[("NO_PROXY", "*")],
    ] {
        assert!(
            proxy_bypass_warning(env_of(pairs), &default_policy).is_none(),
            "{pairs:?}"
        );
    }
    // Nothing is pinned with an allowlist (every host
    // fetched is listed, and listed hosts are exempt),
    // with blocking off, or with the any-source opt-out:
    // every fetch then honours the proxy.
    for pairs in [
        &[("CRABCROP_ALLOWED_SOURCES", "images.example.com")][..],
        &[("CRABCROP_BLOCK_PRIVATE_SOURCES", "0")],
        &[("CRABCROP_ALLOW_ANY_SOURCE", "1")],
    ] {
        let policy = build_source_policy(env_of(pairs)).unwrap();
        assert!(!policy.pins_hostnames(), "{pairs:?}");
        assert!(
            proxy_bypass_warning(proxied("HTTPS_PROXY"), &policy).is_none(),
            "{pairs:?}"
        );
    }
}

#[test]
fn secrets_below_the_minimum_are_rejected() {
    assert!(validate_secret(b"").is_err());
    assert!(validate_secret(b"fifteen-bytes!!").is_err());
    assert!(validate_secret(b"sixteen-bytes!!!").is_ok());
    assert!(validate_secret(b"a much longer production secret").is_ok());
}

#[test]
fn parse_env_value_defaults_parses_and_rejects() {
    assert_eq!(parse_env_value::<u16>("PORT", None, 3000).unwrap(), 3000);
    assert_eq!(
        parse_env_value::<u16>("PORT", Some("8080".to_string()), 3000).unwrap(),
        8080
    );
    let err = parse_env_value::<u16>("PORT", Some("abc".to_string()), 3000)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PORT"), "{err}");
}

#[test]
fn parse_bool_value_accepts_the_spellings_and_rejects_typos() {
    for v in ["1", "true", "TRUE", "yes", "Yes", "on", "ON"] {
        assert!(
            parse_bool_value("CRABCROP_ALLOW_UNSAFE", Some(v.to_string()), false).unwrap(),
            "{v}"
        );
    }
    for v in ["0", "false", "False", "no", "off", "OFF"] {
        assert!(
            !parse_bool_value("CRABCROP_ALLOW_UNSAFE", Some(v.to_string()), true).unwrap(),
            "{v}"
        );
    }
    assert!(!parse_bool_value("X", None, false).unwrap());
    assert!(parse_bool_value("X", None, true).unwrap());
    // A typo is a boot error naming the variable, not a
    // silent false.
    let err = parse_bool_value("CRABCROP_ALLOW_UNSAFE", Some("ture".to_string()), false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("CRABCROP_ALLOW_UNSAFE"), "{err}");
    assert!(err.contains("ture"), "{err}");
}

/// An environment lookup over literal pairs: the
/// construction tests drive the `build_*` functions
/// through this instead of the process environment,
/// so parallel tests cannot race each other.
pub(crate) fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: std::collections::HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |name: &str| map.get(name).cloned()
}

/// The startup error a construction produced, as its
/// message, or a panic: every refusal below is pinned by
/// its message.
pub(crate) fn expect_err<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected a startup error, but construction succeeded"),
    }
}

/// A temp file with the bytes a construction test gave
/// it, in a scratch directory of its own that lives as
/// long as this does.
pub(crate) struct TempFile {
    path: String,
    _dir: crate::test_support::ScratchDir,
}

impl TempFile {
    pub(crate) fn as_str(&self) -> &str {
        &self.path
    }
}

impl AsRef<std::path::Path> for TempFile {
    fn as_ref(&self) -> &std::path::Path {
        self.path.as_ref()
    }
}

/// Both formats print the path alone, as the `String`
/// this replaced did: the detector tests quote it into
/// JSON with `{file:?}` and splice it in with `{file}`.
impl std::fmt::Debug for TempFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.path, f)
    }
}

impl std::fmt::Display for TempFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.path)
    }
}

/// Write bytes to a fresh temp file: construction tests
/// need real files (a presets file, a fallback image, an
/// encoder program) without committing fixtures. Each
/// starts in an empty directory, so no test sees a file
/// an earlier test process left (issue #526). The file
/// is named after its tag too, so two files that must
/// differ only in content (the fingerprint tests) differ
/// in more than their directory.
pub(crate) fn temp_file(tag: &str, contents: &[u8]) -> TempFile {
    let dir = crate::test_support::scratch_dir(&format!("boot-{tag}"));
    let path = dir
        .join(format!("{tag}.tmp"))
        .to_string_lossy()
        .into_owned();
    std::fs::write(&path, contents).unwrap();
    TempFile { path, _dir: dir }
}

/// A path that does not exist: a name inside a scratch
/// directory that is removed again before this returns.
pub(crate) fn missing_path(tag: &str) -> String {
    crate::test_support::scratch_dir(&format!("boot-missing-{tag}"))
        .join("missing.tmp")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn keyring_construction_requires_a_secret_of_at_least_16_bytes() {
    // Unset: a hard error naming the variable.
    let err = expect_err(build_keyring(env_of(&[])));
    assert_eq!(err, "CRABCROP_SECRET must be set");
    // Below the minimum (security audit PL-3): refused.
    let err = expect_err(build_keyring(env_of(&[("CRABCROP_SECRET", "too-short")])));
    assert!(
        err.contains("CRABCROP_SECRET must be at least 16 bytes"),
        "{err}"
    );
    // Valid: the key ID defaults to `current`, no window.
    let keyring =
        build_keyring(env_of(&[("CRABCROP_SECRET", "a production-grade secret")])).unwrap();
    assert_eq!(keyring.current_id, "current");
    assert!(keyring.previous.is_none());
}

#[test]
fn keyring_construction_opens_a_rotation_window() {
    let keyring = build_keyring(env_of(&[
        ("CRABCROP_SECRET", "the current production secret"),
        ("CRABCROP_SECRET_KEY_ID", "2026-10"),
        ("CRABCROP_SECRET_PREVIOUS", "the outgoing production secret"),
        ("CRABCROP_SECRET_PREVIOUS_KEY_ID", "2026-06"),
    ]))
    .unwrap();
    assert_eq!(keyring.current_id, "2026-10");
    let (id, _) = keyring.previous.as_ref().unwrap();
    assert_eq!(id, "2026-06");
    // The previous key ID defaults to `previous`.
    let keyring = build_keyring(env_of(&[
        ("CRABCROP_SECRET", "the current production secret"),
        ("CRABCROP_SECRET_PREVIOUS", "the outgoing production secret"),
    ]))
    .unwrap();
    assert_eq!(keyring.previous.as_ref().unwrap().0, "previous");
    // The previous secret is held to the same minimum.
    let err = expect_err(build_keyring(env_of(&[
        ("CRABCROP_SECRET", "the current production secret"),
        ("CRABCROP_SECRET_PREVIOUS", "short"),
    ])));
    assert!(
        err.contains("CRABCROP_SECRET_PREVIOUS must be at least 16 bytes"),
        "{err}"
    );
    // The two IDs must differ, or rotation routing is ambiguous.
    let err = expect_err(build_keyring(env_of(&[
        ("CRABCROP_SECRET", "the current production secret"),
        ("CRABCROP_SECRET_PREVIOUS", "the outgoing production secret"),
        ("CRABCROP_SECRET_PREVIOUS_KEY_ID", "current"),
    ])));
    assert!(err.contains("must differ"), "{err}");
    // A malformed key ID is refused, naming the variable.
    let err = expect_err(build_keyring(env_of(&[
        ("CRABCROP_SECRET", "the current production secret"),
        ("CRABCROP_SECRET_KEY_ID", "not a key id!"),
    ])));
    assert!(err.contains("CRABCROP_SECRET_KEY_ID"), "{err}");
}

#[test]
fn source_config_construction_enforces_pairing_and_the_bucket_policy() {
    // Half a credential pair is a startup error.
    let err = expect_err(build_source_config(env_of(&[(
        "AWS_ACCESS_KEY_ID",
        "AKID",
    )])));
    assert!(err.contains("must be set together"), "{err}");
    let err = expect_err(build_source_config(env_of(&[(
        "CRABCROP_GCS_SECRET_KEY",
        "s",
    )])));
    assert!(err.contains("must be set together"), "{err}");
    // Full credentials with no allowlist: fail closed
    // (security audit L-1) at construction time.
    let err = expect_err(build_source_config(env_of(&[
        ("AWS_ACCESS_KEY_ID", "AKID"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
    ])));
    assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
    // Credentials plus an allowlist construct fine, and
    // the session token is attached to the pair.
    let sources = build_source_config(env_of(&[
        ("AWS_ACCESS_KEY_ID", "AKID"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
        ("AWS_SESSION_TOKEN", "token"),
        ("CRABCROP_ALLOWED_BUCKETS", "photos, backups"),
    ]))
    .unwrap();
    assert_eq!(sources.allowed_buckets, ["photos", "backups"]);
    let creds = sources.aws_creds.as_ref().unwrap();
    assert_eq!(creds.session_token.as_deref(), Some("token"));
    // Nothing configured also constructs: unsigned
    // fetches, every bucket allowed.
    let sources = build_source_config(env_of(&[])).unwrap();
    assert!(sources.aws_creds.is_none() && sources.allowed_buckets.is_empty());
}

#[test]
fn source_config_construction_validates_endpoints_and_the_file_root() {
    // Both store endpoints take the same check (issue
    // #455: the GCS one used to boot with any scheme):
    // a non-http(s) URL is refused, naming the variable,
    // and so is a value that is not a URL at all.
    for name in ["CRABCROP_S3_ENDPOINT", "CRABCROP_GCS_ENDPOINT"] {
        for bad in [
            "ftp://storage.example",
            "file:///etc/passwd",
            "gs://bucket",
            "s3://bucket",
            "data:text/plain,x",
            "javascript:alert(1)",
            "unix:/var/run/store.sock",
        ] {
            let err = expect_err(build_source_config(env_of(&[(name, bad)])));
            assert_eq!(err, format!("{name} must be an http(s) URL"), "{bad}");
        }
        for bad in [
            "storage.example",
            "//storage.example",
            "http://",
            "http://[::1",
            " ",
        ] {
            let err = expect_err(build_source_config(env_of(&[(name, bad)])));
            assert!(
                err.starts_with(&format!("invalid {name}: ")),
                "{bad}: {err}"
            );
        }
    }
    // A valid endpoint is stored with the trailing
    // slash trimmed.
    let sources = build_source_config(env_of(&[
        ("CRABCROP_S3_ENDPOINT", "http://127.0.0.1:9000/"),
        ("CRABCROP_GCS_ENDPOINT", "https://gcs.example/"),
    ]))
    .unwrap();
    assert_eq!(
        sources.s3_endpoint.as_deref(),
        Some("http://127.0.0.1:9000")
    );
    assert_eq!(sources.gcs_endpoint, "https://gcs.example");
    // A file root that does not exist is refused...
    let missing = missing_path("root");
    let err = expect_err(build_source_config(env_of(&[(
        "CRABCROP_FILE_ROOT",
        missing.as_str(),
    )])));
    assert!(err.contains("CRABCROP_FILE_ROOT is not usable"), "{err}");
    // ...as is one that is not a directory.
    let file = temp_file("root-file", b"not a directory");
    let err = expect_err(build_source_config(env_of(&[(
        "CRABCROP_FILE_ROOT",
        file.as_str(),
    )])));
    assert!(err.contains("is not a directory"), "{err}");
    // A real directory is canonicalized into the config.
    let dir = std::env::temp_dir().to_string_lossy().into_owned();
    let sources = build_source_config(env_of(&[("CRABCROP_FILE_ROOT", dir.as_str())])).unwrap();
    assert!(sources.file_root.is_some());
}

#[test]
fn upload_store_construction_follows_the_destination_rules() {
    let plain = SourceConfig::default();
    // Built where the temp dir is, never at a fixed
    // /tmp: TMPDIR moves it (issue #526).
    let scratch = crate::test_support::scratch_dir("boot-upload");
    let root_dir = scratch.join("root");
    std::fs::create_dir(&root_dir).unwrap();
    let root = root_dir.to_string_lossy().into_owned();
    let uploads = root_dir.join("uploads").to_string_lossy().into_owned();
    // Its name starts with the root's, and it is not
    // under the root.
    let sibling = scratch.join("root-other").to_string_lossy().into_owned();
    // Neither destination: uploads are off, not an error.
    assert!(build_upload_store(env_of(&[]), &plain).unwrap().is_none());
    // Both destinations: a startup error.
    let err = expect_err(build_upload_store(
        env_of(&[
            ("CRABCROP_UPLOAD_DIR", uploads.as_str()),
            ("CRABCROP_UPLOAD_BUCKET", "uploads"),
        ]),
        &plain,
    ));
    assert!(err.contains("exactly one upload destination"), "{err}");
    // A local destination constructs with the defaults:
    // 10 MB cap (the source cap), overwrite off.
    let store = build_upload_store(env_of(&[("CRABCROP_UPLOAD_DIR", uploads.as_str())]), &plain)
        .unwrap()
        .unwrap();
    assert_eq!(store.max_bytes, 10 * 1024 * 1024);
    assert!(!store.allow_overwrite);
    assert!(store.local_dir().is_some());
    // The cap may only move down from the source cap.
    let err = expect_err(build_upload_store(
        env_of(&[
            ("CRABCROP_UPLOAD_DIR", uploads.as_str()),
            ("CRABCROP_UPLOAD_MAX_MB", "11"),
        ]),
        &plain,
    ));
    assert!(err.contains("CRABCROP_UPLOAD_MAX_MB"), "{err}");
    let store = build_upload_store(
        env_of(&[
            ("CRABCROP_UPLOAD_DIR", uploads.as_str()),
            ("CRABCROP_UPLOAD_MAX_MB", "2"),
            ("CRABCROP_UPLOAD_ALLOW_OVERWRITE", "true"),
        ]),
        &plain,
    )
    .unwrap()
    .unwrap();
    assert_eq!(store.max_bytes, 2 * 1024 * 1024);
    assert!(store.allow_overwrite);
    // A local destination outside a configured file
    // root is refused: its objects could never serve.
    let rooted = build_source_config(env_of(&[("CRABCROP_FILE_ROOT", root.as_str())])).unwrap();
    // The root itself and a directory inside it are under
    // it, and their objects are named from the root.
    for (dir, object) in [
        (root.as_str(), "file:///a.jpg"),
        (uploads.as_str(), "file:///uploads/a.jpg"),
    ] {
        let store = build_upload_store(env_of(&[("CRABCROP_UPLOAD_DIR", dir)]), &rooted)
            .unwrap()
            .unwrap();
        assert_eq!(store.source_ref("a.jpg"), object, "{dir}");
    }
    // The check compares path components, not strings:
    // the sibling is refused like `/`.
    for outside in ["/", sibling.as_str()] {
        let err = expect_err(build_upload_store(
            env_of(&[("CRABCROP_UPLOAD_DIR", outside)]),
            &rooted,
        ));
        assert!(err.contains("CRABCROP_FILE_ROOT"), "{outside}: {err}");
    }
    // A bucket destination with credentials must be in
    // the readable allowlist, or its objects could
    // never serve back.
    let signed = build_source_config(env_of(&[
        ("AWS_ACCESS_KEY_ID", "AKID"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
        ("CRABCROP_ALLOWED_BUCKETS", "photos"),
    ]))
    .unwrap();
    let err = expect_err(build_upload_store(
        env_of(&[("CRABCROP_UPLOAD_BUCKET", "uploads")]),
        &signed,
    ));
    assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
    let store = build_upload_store(
        env_of(&[
            ("CRABCROP_UPLOAD_BUCKET", "photos"),
            ("CRABCROP_UPLOAD_PREFIX", "orig"),
        ]),
        &signed,
    )
    .unwrap()
    .unwrap();
    assert_eq!(store.source_ref("a/b.jpg"), "s3://photos/orig/a/b.jpg");
    // The same rule without credentials: bucket loads
    // require an allowlist entry either way, so an
    // upload bucket outside the list could never serve
    // back.
    let err = expect_err(build_upload_store(
        env_of(&[("CRABCROP_UPLOAD_BUCKET", "uploads")]),
        &plain,
    ));
    assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
}

#[test]
fn the_upload_cap_follows_the_source_fetch_cap() {
    const MB: u64 = 1024 * 1024;
    let plain = SourceConfig::default();
    let scratch = crate::test_support::scratch_dir("boot-upload-cap");
    let dir = scratch.to_string_lossy().into_owned();
    let upload = |pairs: &[(&str, &str)]| {
        let mut all = vec![("CRABCROP_UPLOAD_DIR", dir.as_str())];
        all.extend_from_slice(pairs);
        build_upload_store(env_of(&all), &plain)
    };
    // A raised fetch cap (issue #471) raises the upload
    // cap's default and its maximum with it.
    let store = upload(&[("CRABCROP_SOURCE_MAX_MB", "64")])
        .unwrap()
        .unwrap();
    assert_eq!(store.max_bytes, 64 * MB);
    let err = expect_err(upload(&[
        ("CRABCROP_SOURCE_MAX_MB", "64"),
        ("CRABCROP_UPLOAD_MAX_MB", "65"),
    ]));
    assert!(err.contains("between 1 and 64"), "{err}");
    // A lowered one lowers both: an upload cap the
    // default allowed is refused, so a stored original
    // can never outgrow what the loader fetches back.
    let store = upload(&[("CRABCROP_SOURCE_MAX_MB", "2")]).unwrap().unwrap();
    assert_eq!(store.max_bytes, 2 * MB);
    let err = expect_err(upload(&[
        ("CRABCROP_SOURCE_MAX_MB", "2"),
        ("CRABCROP_UPLOAD_MAX_MB", "10"),
    ]));
    assert!(err.contains("CRABCROP_UPLOAD_MAX_MB"), "{err}");
    assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{err}");
    // A bad fetch cap fails the build, naming the cap.
    let err = expect_err(upload(&[("CRABCROP_SOURCE_MAX_MB", "2048")]));
    assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{err}");
}

#[test]
fn presets_construction_reads_and_validates_the_file() {
    // Nothing configured: no presets, free-form allowed.
    let (presets, only) = build_presets(env_of(&[])).unwrap();
    assert!(presets.is_empty() && !only);
    // A missing file is a startup error naming the variable.
    let missing = missing_path("presets");
    let err = expect_err(build_presets(env_of(&[(
        "CRABCROP_PRESETS_FILE",
        missing.as_str(),
    )])));
    assert!(err.contains("cannot read CRABCROP_PRESETS_FILE"), "{err}");
    // Malformed content is refused at boot...
    let bad = temp_file("presets-bad", b"this line has no equals\n");
    let err = expect_err(build_presets(env_of(&[(
        "CRABCROP_PRESETS_FILE",
        bad.as_str(),
    )])));
    assert!(err.contains("not `name = operations`"), "{err}");
    // ...and so are operations that are not a preset body.
    let bad_ops = temp_file("presets-bad-ops", b"thumb = preset:other\n");
    assert!(build_presets(env_of(&[("CRABCROP_PRESETS_FILE", bad_ops.as_str())])).is_err());
    // A valid file loads, and presets-only is honored with it...
    let good = temp_file("presets-good", b"thumb = 300x200\n");
    let (presets, only) = build_presets(env_of(&[
        ("CRABCROP_PRESETS_FILE", good.as_str()),
        ("CRABCROP_ONLY_PRESETS", "1"),
    ]))
    .unwrap();
    assert_eq!(presets["thumb"].as_str(), "300x200");
    assert!(only);
    // ...but refused without one: it would reject every
    // transformed request.
    let err = expect_err(build_presets(env_of(&[("CRABCROP_ONLY_PRESETS", "true")])));
    assert!(err.contains("defines no presets"), "{err}");
    // A typo'd flag is an error, not a silent false.
    let err = expect_err(build_presets(env_of(&[("CRABCROP_ONLY_PRESETS", "ture")])));
    assert!(err.contains("CRABCROP_ONLY_PRESETS"), "{err}");
}

#[test]
fn source_roots_construction_reads_and_validates_the_file() {
    // Nothing configured: no roots.
    assert!(build_source_roots(env_of(&[])).unwrap().is_empty());
    // A missing file is a startup error naming the variable.
    let missing = missing_path("roots");
    let err = expect_err(build_source_roots(env_of(&[(
        "CRABCROP_SOURCE_ROOTS_FILE",
        missing.as_str(),
    )])));
    assert!(
        err.contains("cannot read CRABCROP_SOURCE_ROOTS_FILE"),
        "{err}"
    );
    // A valid file loads, comments and blank lines
    // skipped, bases stored in parsed form with the
    // trailing slash trimmed.
    let good = temp_file(
            "roots-good",
            b"# origins\n\nphotos = https://cdn.example.com/photos/\nmedia = http://assets.example.com:8080\n",
        );
    let roots =
        build_source_roots(env_of(&[("CRABCROP_SOURCE_ROOTS_FILE", good.as_str())])).unwrap();
    assert_eq!(roots["photos"], "https://cdn.example.com/photos");
    assert_eq!(roots["media"], "http://assets.example.com:8080");
    // Malformed content is refused at boot, line by line.
    for (tag, text) in [
        ("roots-noeq", "photos https://cdn.example.com\n"),
        ("roots-badname", "my photos = https://cdn.example.com\n"),
        (
            "roots-dup",
            "a = https://x.example.com\na = https://y.example.com\n",
        ),
        ("roots-noturl", "a = cdn.example.com/photos\n"),
        ("roots-scheme", "a = s3://bucket/key\n"),
        ("roots-userinfo", "a = https://user:pass@cdn.example.com\n"),
        ("roots-query", "a = https://cdn.example.com/?v=1\n"),
        ("roots-fragment", "a = https://cdn.example.com/#top\n"),
        // Issue #250: the name charset and the base
        // shape admit no lookalikes either.
        ("roots-dotname", "my.photos = https://cdn.example.com\n"),
        ("roots-uniname", "fötös = https://cdn.example.com\n"),
        ("roots-emptyname", "= https://cdn.example.com\n"),
        ("roots-ftp", "a = ftp://cdn.example.com\n"),
        ("roots-useronly", "a = https://user@cdn.example.com\n"),
        ("roots-nohost", "a = https://\n"),
    ] {
        let path = temp_file(tag, text.as_bytes());
        assert!(
            build_source_roots(env_of(&[("CRABCROP_SOURCE_ROOTS_FILE", path.as_str())])).is_err(),
            "{tag}"
        );
    }
}

#[test]
fn the_source_buffer_budget_scales_with_the_lanes_and_checks_its_bounds() {
    const MB: u64 = 1024 * 1024;
    // Unset: four fetch caps per lane permit, so a
    // one-core host with the default slow lane (two
    // permits) gets 80 MB and a four-core one 200 MB.
    assert_eq!(source_buffer_budget(env_of(&[]), 2).unwrap(), 80 * MB);
    assert_eq!(source_buffer_budget(env_of(&[]), 5).unwrap(), 200 * MB);
    // The default never leaves the accepted range.
    assert_eq!(
        source_buffer_budget(env_of(&[]), usize::MAX).unwrap(),
        MAX_SOURCE_BUFFER_MB * MB
    );
    // Set: megabytes, with one fetch cap as the floor.
    let set = |v: &str| source_buffer_budget(env_of(&[("CRABCROP_SOURCE_BUFFER_MAX_MB", v)]), 2);
    assert_eq!(set("10").unwrap(), 10 * MB);
    assert_eq!(set("65536").unwrap(), 65_536 * MB);
    for bad in ["0", "9", "65537", "lots", "-1", ""] {
        let err = expect_err(set(bad));
        assert!(
            err.contains("CRABCROP_SOURCE_BUFFER_MAX_MB"),
            "{bad}: {err}"
        );
    }
}

#[test]
fn source_fetch_bounds_default_to_the_loader_and_check_their_ranges() {
    // Unset: the loader's own defaults, 10 MiB and 10 s.
    assert_eq!(source_max_mb(&env_of(&[])).unwrap(), 10);
    assert_eq!(source_timeout_secs(&env_of(&[])).unwrap(), 10);
    let cap = |v: &str| source_max_mb(&env_of(&[("CRABCROP_SOURCE_MAX_MB", v)]));
    let timeout = |v: &str| source_timeout_secs(&env_of(&[("CRABCROP_SOURCE_TIMEOUT_SECS", v)]));
    // The range edges are accepted.
    assert_eq!(cap("1").unwrap(), 1);
    assert_eq!(cap("1024").unwrap(), 1024);
    assert_eq!(timeout("1").unwrap(), 1);
    assert_eq!(timeout("600").unwrap(), 600);
    // Out of range, unparseable, and hostile spellings
    // are boot errors naming the variable, never a
    // silent default.
    for bad in [
        "0",
        "1025",
        "-1",
        "10.5",
        " 10",
        "10MB",
        "",
        "lots",
        "18446744073709551616",
        "1\u{0}",
    ] {
        let err = expect_err(cap(bad));
        assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{bad:?}: {err}");
    }
    for bad in [
        "0",
        "601",
        "-1",
        "1.5",
        "10s",
        "",
        "soon",
        "18446744073709551616",
    ] {
        let err = expect_err(timeout(bad));
        assert!(
            err.contains("CRABCROP_SOURCE_TIMEOUT_SECS"),
            "{bad:?}: {err}"
        );
    }
}

#[test]
fn the_shutdown_grace_defaults_under_ten_seconds_and_checks_its_range() {
    // Issue #463: unset, the drain ends before the 10 s
    // `docker stop` and Cloud Run allow.
    let grace = shutdown_grace_secs(&env_of(&[])).unwrap();
    assert_eq!(grace, crate::server::DEFAULT_SHUTDOWN_GRACE_SECS);
    assert!(grace < 10, "{grace}");
    let set = |v: &str| shutdown_grace_secs(&env_of(&[("CRABCROP_SHUTDOWN_GRACE_SECS", v)]));
    assert_eq!(set("1").unwrap(), 1);
    assert_eq!(set("600").unwrap(), 600);
    for bad in ["0", "601", "-1", "2.5", "8s", "", "never"] {
        let err = expect_err(set(bad));
        assert!(
            err.contains("CRABCROP_SHUTDOWN_GRACE_SECS"),
            "{bad:?}: {err}"
        );
    }
}

#[test]
fn the_source_buffer_budget_follows_the_configured_fetch_cap() {
    const MB: u64 = 1024 * 1024;
    let budget = |pairs: &[(&str, &str)], permits| source_buffer_budget(env_of(pairs), permits);
    // Unset: four fetch caps per lane permit at the
    // configured cap (issue #471), up or down.
    assert_eq!(
        budget(&[("CRABCROP_SOURCE_MAX_MB", "64")], 2).unwrap(),
        512 * MB
    );
    assert_eq!(
        budget(&[("CRABCROP_SOURCE_MAX_MB", "1")], 2).unwrap(),
        8 * MB
    );
    // The largest cap on a large host stays inside the
    // accepted range.
    assert_eq!(
        budget(&[("CRABCROP_SOURCE_MAX_MB", "1024")], 64).unwrap(),
        MAX_SOURCE_BUFFER_MB * MB
    );
    // Set: the floor is one configured cap, so a budget
    // that could never hold a source at the cap is
    // refused, and a lower cap lowers the floor.
    let err = expect_err(budget(
        &[
            ("CRABCROP_SOURCE_MAX_MB", "64"),
            ("CRABCROP_SOURCE_BUFFER_MAX_MB", "63"),
        ],
        2,
    ));
    assert!(err.contains("between 64"), "{err}");
    assert_eq!(
        budget(
            &[
                ("CRABCROP_SOURCE_MAX_MB", "64"),
                ("CRABCROP_SOURCE_BUFFER_MAX_MB", "64"),
            ],
            2
        )
        .unwrap(),
        64 * MB
    );
    assert_eq!(
        budget(
            &[
                ("CRABCROP_SOURCE_MAX_MB", "1"),
                ("CRABCROP_SOURCE_BUFFER_MAX_MB", "1"),
            ],
            2
        )
        .unwrap(),
        MB
    );
    // A bad cap fails the budget too, naming the cap.
    let err = expect_err(budget(&[("CRABCROP_SOURCE_MAX_MB", "0")], 2));
    assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{err}");
}

#[test]
fn parses_detector_declarations() {
    let decls = parse_detector_declarations(
        r#"[
              {"name": "crates", "model": "/models/crates.onnx", "kind": "boxes",
               "labels": "/models/crates.txt", "threshold": 0.4},
              {"name": "logos", "model": "/models/logos.onnx", "kind": "boxes",
               "labels": ["nike", "adidas"]},
              {"name": "saliency", "model": "/models/saliency.onnx", "kind": "focal"}
            ]"#,
    )
    .unwrap();
    assert_eq!(decls.len(), 3);
    assert_eq!(decls[0].kind, DetectorKind::Boxes);
    assert_eq!(
        decls[0].labels,
        Some(LabelsDecl::Path("/models/crates.txt".to_string()))
    );
    assert_eq!(decls[0].threshold, Some(0.4));
    assert_eq!(
        decls[1].labels,
        Some(LabelsDecl::Inline(vec![
            "nike".to_string(),
            "adidas".to_string()
        ]))
    );
    assert_eq!(decls[1].threshold, None);
    assert_eq!(decls[2].kind, DetectorKind::Focal);
    assert_eq!(decls[2].labels, None);
    // An empty file declares nothing, validly.
    assert!(parse_detector_declarations("[]").unwrap().is_empty());
}

#[test]
fn detector_declaration_refusals_name_the_declaration() {
    // Every refusal class that precedes model contact,
    // each pinned by its wording (issue #166).
    let cases: &[(&str, &str)] = &[
        ("not json", "not valid JSON"),
        ("{}", "must hold a JSON array"),
        (r#"["crates"]"#, "is not a JSON object"),
        (
            r#"[{"model": "/m.onnx", "kind": "boxes"}]"#,
            "\"name\" must be a non-empty string",
        ),
        (
            r#"[{"name": "bad name!", "model": "/m.onnx", "kind": "boxes"}]"#,
            "invalid detector name",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes"},
                    {"name": "a", "model": "/n.onnx", "kind": "focal"}]"#,
            "duplicate detector name: a",
        ),
        (
            r#"[{"name": "a", "kind": "boxes"}]"#,
            "\"model\" must be a non-empty path string",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx"}]"#,
            "\"kind\" must be \"boxes\" or \"focal\"",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "heatmap"}]"#,
            "unknown kind \"heatmap\"",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "extra": 1}]"#,
            "unknown field \"extra\"",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "focal", "labels": "/l.txt"}]"#,
            "\"labels\" applies only to \"boxes\" detectors",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "focal", "threshold": 0.4}]"#,
            "\"threshold\" applies only to \"boxes\" detectors",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": 42}]"#,
            "\"labels\" must be a file path string or an array of class names",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": []}]"#,
            "\"labels\" must name at least one class",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": ["ok", " "]}]"#,
            "\"labels\" entries must be non-empty strings",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "threshold": "high"}]"#,
            "\"threshold\" must be a number",
        ),
        (
            r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "threshold": 1.5}]"#,
            "\"threshold\" must be within 0..=1",
        ),
    ];
    for (text, wording) in cases {
        let err = expect_err(parse_detector_declarations(text));
        assert!(err.contains(wording), "{text}: {err}");
    }
    // The unknown-kind refusal names the declaration.
    let err = expect_err(parse_detector_declarations(
        r#"[{"name": "crates", "model": "/m.onnx", "kind": "heatmap"}]"#,
    ));
    assert!(err.contains("\"crates\""), "{err}");
}

#[test]
fn detector_declarations_enforce_resource_limits() {
    // The caps from audit R2-27: declaration count,
    // label count and length, and the reserved name.
    let decl = |i: usize| format!(r#"{{"name": "d{i}", "model": "/m.onnx", "kind": "focal"}}"#);
    let many = (0..=MAX_DETECTOR_DECLARATIONS)
        .map(decl)
        .collect::<Vec<_>>()
        .join(",");
    let err = expect_err(parse_detector_declarations(&format!("[{many}]")));
    assert!(err.contains("the cap is"), "{err}");
    // Exactly at the cap parses.
    let at_cap = (0..MAX_DETECTOR_DECLARATIONS)
        .map(decl)
        .collect::<Vec<_>>()
        .join(",");
    assert_eq!(
        parse_detector_declarations(&format!("[{at_cap}]"))
            .unwrap()
            .len(),
        MAX_DETECTOR_DECLARATIONS
    );
    // `builtin` is reserved for the built-in detector's
    // meta attribution.
    let err = expect_err(parse_detector_declarations(
        r#"[{"name": "builtin", "model": "/m.onnx", "kind": "boxes"}]"#,
    ));
    assert!(err.contains("reserved"), "{err}");
    // Inline labels: one over the count cap refuses, at
    // the cap parses, and an overlong label refuses.
    let labels = |n: usize| {
        (0..n)
            .map(|i| format!("\"c{i}\""))
            .collect::<Vec<_>>()
            .join(",")
    };
    let over = format!(
        r#"[{{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": [{}]}}]"#,
        labels(MAX_DETECTOR_LABELS + 1)
    );
    let err = expect_err(parse_detector_declarations(&over));
    assert!(err.contains("the cap is"), "{err}");
    let at = format!(
        r#"[{{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": [{}]}}]"#,
        labels(MAX_DETECTOR_LABELS)
    );
    assert!(parse_detector_declarations(&at).is_ok());
    let long_label = "x".repeat(MAX_DETECTOR_LABEL_CHARS + 1);
    let err = expect_err(parse_detector_declarations(&format!(
        r#"[{{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": ["{long_label}"]}}]"#
    )));
    assert!(err.contains("at most"), "{err}");
}

#[test]
fn parses_preset_files() {
    let map = parse_presets(
            "# cards\nog-card = crop:1200x630:smart/filters:format(webp):quality(80)\n\nthumb = 300x200\n",
        )
        .unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map["thumb"].as_str(), "300x200");
    for bad in [
        "no-equals-here",
        "bad name = 300x200",
        "thumb = 300x200\nthumb = 100x100",
        "thumb = preset:other",
        "thumb = example.com/img.jpg",
        "thumb = ",
    ] {
        assert!(parse_presets(bad).is_err(), "{bad}");
    }
    // A body over the operation cap refuses at boot
    // (issue #268): at the cap it parses, one over it
    // does not.
    let at_cap = format!("big = {}", vec!["300x200"; MAX_PRESET_OPERATIONS].join("/"));
    assert!(parse_presets(&at_cap).is_ok());
    let over_cap = format!(
        "big = {}",
        vec!["300x200"; MAX_PRESET_OPERATIONS + 1].join("/")
    );
    let err = parse_presets(&over_cap).unwrap_err().to_string();
    assert!(err.contains("operation cap"), "{err}");
    // The count is the one requests are held to (issue
    // #469): each filter counts, so one `filters:`
    // segment carrying 101 filters is over the cap too.
    let filters = |n: usize| format!("big = filters:{}", vec!["invert"; n].join(":"));
    assert!(parse_presets(&filters(MAX_PRESET_OPERATIONS)).is_ok());
    let err = parse_presets(&filters(MAX_PRESET_OPERATIONS + 1))
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "preset big has 101 operations, over the 100 operation cap"
    );
    // A body longer than a request may expand to refuses
    // at boot too: one operation, but a long watermark URL.
    let long = |len: usize| {
        let (head, tail) = (
            "filters:watermark(https%3A%2F%2Fexample.com%2F",
            ".png,0,0,100)",
        );
        format!("{head}{}{tail}", "w".repeat(len - head.len() - tail.len()))
    };
    assert_eq!(long(MAX_PRESET_LEN).len(), MAX_PRESET_LEN);
    let map = parse_presets(&format!("big = {}", long(MAX_PRESET_LEN))).unwrap();
    assert_eq!(map["big"].as_str().len(), MAX_PRESET_LEN);
    let err = parse_presets(&format!("big = {}", long(MAX_PRESET_LEN + 1)))
        .unwrap_err()
        .to_string();
    assert_eq!(err, "preset big is 262145 bytes, over the 262144 byte cap");
}

/// The secret every `Config::from_env` test boots with.
const SECRET: &str = "a production-grade secret";

/// `Config::from_env` over the secret plus `pairs`.
fn config_of(pairs: &[(&str, &str)]) -> Result<Config, ConfigError> {
    let mut all = vec![("CRABCROP_SECRET", SECRET)];
    all.extend_from_slice(pairs);
    Config::from_env(env_of(&all))
}

/// The refusal `Config::from_env` made of `pairs`, or a panic.
fn refusal(pairs: &[(&str, &str)]) -> ConfigError {
    match config_of(pairs) {
        Err(e) => e,
        Ok(_) => panic!("expected {pairs:?} to be refused"),
    }
}

#[test]
fn from_env_reads_every_default() {
    let config = config_of(&[]).unwrap();
    assert_eq!(config.keyring.current_id, "current");
    assert!(!config.allow_unsafe);
    assert_eq!(config.port, 3000);
    assert_eq!(config.ssim_target, crate::quality::DEFAULT_SSIM_TARGET);
    assert!(config.metrics_token.is_none());
    assert_eq!(config.process_timeout_secs, 30);
    assert_eq!(
        config.slow_lane_permits,
        crate::server::DEFAULT_SLOW_LANE_PERMITS
    );
    assert_eq!(config.fetch_cap_mb, 10);
    assert_eq!(config.fetch_timeout_secs, 10);
    assert_eq!(
        config.shutdown_grace_secs,
        crate::server::DEFAULT_SHUTDOWN_GRACE_SECS
    );
    let lanes = (config.process_permits + config.slow_lane_permits) as u64;
    assert_eq!(
        config.source_buffer_bytes,
        (40 * lanes).clamp(10, 65_536) * 1024 * 1024
    );
    assert_eq!(config.avif_speed, crate::engine::DEFAULT_AVIF_SPEED);
    assert_eq!(config.width_step, DEFAULT_WIDTH_STEP);
    assert_eq!(config.cache_bytes, 256 * 1024 * 1024);
    assert!(config.policy.block_private);
    assert!(config.upload.is_none());
    assert!(config.presets.is_empty() && !config.only_presets);
    assert!(config.source_roots.is_empty() && config.default_ops.is_empty());
    assert!(!config.preserve_metadata && config.brand_color.is_none());
    assert!(config.result_storage.is_none() && config.result_bucket.is_none());
    assert!(config.ai.clip.is_none() && config.ai.detectors.is_empty());
    assert!(config.font_path.is_none() && config.fallback_image.is_none());
    assert_eq!(
        config.video.timeout_secs,
        crate::video::DEFAULT_TIMEOUT_SECS
    );
    assert_eq!(config.video.max_mb, crate::video::DEFAULT_MAX_OUTPUT_MB);
    assert!(config.video.encoder.is_none() && config.pdf.renderer.is_none());
    assert!(config.provenance.is_none() && !config.meta_enabled);
    // The fingerprint is the one the boot defaults give.
    assert_eq!(
        config.config_fingerprint,
        crate::config_fingerprint(
            env_of(&[]),
            false,
            crate::quality::DEFAULT_SSIM_TARGET,
            crate::engine::DEFAULT_AVIF_SPEED,
            crate::engine::DEFAULT_JXL_SPEED,
            DEFAULT_WIDTH_STEP,
        )
    );
}

#[test]
fn from_env_carries_what_it_read() {
    let config = config_of(&[
        ("CRABCROP_PORT", "8080"),
        ("PORT", "9090"),
        ("CRABCROP_METRICS_TOKEN", "metrics-token"),
        ("CRABCROP_RESULT_STORAGE_DIR", "/var/cache/crabcrop"),
        ("CRABCROP_RESULT_STORAGE_MAX_MB", "64"),
        ("CRABCROP_RESULT_BUCKET", "results"),
        ("CRABCROP_RESULT_PREFIX", "edge"),
        ("CRABCROP_FONT_PATH", "/fonts/a.ttf"),
        ("CRABCROP_FALLBACK_IMAGE", "https://example.com/f.png"),
        ("CRABCROP_VIDEO_ENCODER", "/bin/ffmpeg"),
        ("CRABCROP_PDF_RENDERER", "/bin/pdftocairo"),
        ("CRABCROP_PDF_TIMEOUT_SECS", "45"),
        ("CRABCROP_C2PA_CERT", "/c2pa/cert.pem"),
        ("CRABCROP_C2PA_KEY", "/c2pa/key.pem"),
        ("CRABCROP_META_MODE", "on"),
    ])
    .unwrap();
    // CRABCROP_PORT wins over PORT.
    assert_eq!(config.port, 8080);
    assert_eq!(
        config.metrics_token.as_ref().map(Secret::expose),
        Some("metrics-token")
    );
    let storage = config.result_storage.unwrap();
    assert_eq!(
        (storage.dir.as_str(), storage.max_mb, storage.max_bytes),
        ("/var/cache/crabcrop", 64, 64 * 1024 * 1024)
    );
    let bucket = config.result_bucket.unwrap();
    assert_eq!(
        (bucket.bucket.as_str(), bucket.prefix.as_str()),
        ("results", "edge")
    );
    assert_eq!(config.font_path.as_deref(), Some("/fonts/a.ttf"));
    assert_eq!(
        config.fallback_image.as_deref(),
        Some("https://example.com/f.png")
    );
    assert_eq!(config.video.encoder.as_deref(), Some("/bin/ffmpeg"));
    assert_eq!(config.pdf.timeout_secs, 45);
    let provenance = config.provenance.unwrap();
    assert_eq!(
        (provenance.cert.as_str(), provenance.key.as_str()),
        ("/c2pa/cert.pem", "/c2pa/key.pem")
    );
    assert!(config.meta_enabled);
}

#[test]
fn from_env_refusals_are_typed_and_keep_the_boot_text() {
    // Missing: the one variable boot cannot do without.
    let err = Config::from_env(env_of(&[])).map(|_| ()).unwrap_err();
    assert!(matches!(
        err,
        ConfigError::Missing {
            var: "CRABCROP_SECRET"
        }
    ));
    assert_eq!(err.to_string(), "CRABCROP_SECRET must be set");
    // Invalid: the variable that was read, and its value.
    let err = refusal(&[("PORT", "abc")]);
    match &err {
        ConfigError::Invalid { var, value, .. } => {
            assert_eq!(*var, "PORT");
            assert_eq!(*value, ConfigValue::Text("abc".to_string()));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(err.to_string(), "invalid PORT: \"abc\"");
    // The alias is named when it is the one read.
    let err = refusal(&[("CRABCROP_PORT", ""), ("PORT", "8080")]);
    assert!(matches!(
        err,
        ConfigError::Invalid {
            var: "CRABCROP_PORT",
            ..
        }
    ));
    assert_eq!(err.to_string(), "invalid CRABCROP_PORT: \"\"");
    let err = refusal(&[("CRABCROP_SSIM_TARGET", "101")]);
    assert_eq!(
        err.to_string(),
        "CRABCROP_SSIM_TARGET must be between 0 and 100, got 101"
    );
    // Conflict: every variable the message names.
    let err = refusal(&[
        ("CRABCROP_UPLOAD_DIR", "/srv/uploads"),
        ("CRABCROP_UPLOAD_BUCKET", "uploads"),
    ]);
    match &err {
        ConfigError::Conflict { vars, .. } => {
            assert_eq!(vars, &["CRABCROP_UPLOAD_DIR", "CRABCROP_UPLOAD_BUCKET"]);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        err.to_string(),
        "CRABCROP_UPLOAD_DIR and CRABCROP_UPLOAD_BUCKET are both set: \
         configure exactly one upload destination"
    );
    let err = refusal(&[("CRABCROP_OCR_DICT", "/m/dict.txt")]);
    assert!(matches!(
        &err,
        ConfigError::Conflict { vars, .. }
            if vars == &["CRABCROP_OCR_DETECT_MODEL", "CRABCROP_OCR_RECOGNIZE_MODEL"]
    ));
    // Io: a configuration file that cannot be read, with
    // the cause as the error's source.
    let missing = missing_path("presets");
    let err = refusal(&[("CRABCROP_PRESETS_FILE", missing.as_str())]);
    match &err {
        ConfigError::Io { var, path, source } => {
            assert_eq!(*var, "CRABCROP_PRESETS_FILE");
            assert_eq!(path, std::path::Path::new(&missing));
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("{other:?}"),
    }
    let text = err.to_string();
    assert!(
        text.starts_with(&format!("cannot read CRABCROP_PRESETS_FILE ({missing}): ")),
        "{text}"
    );
    let source = std::error::Error::source(&err).expect("the I/O error is the source");
    assert!(text.ends_with(&source.to_string()), "{text}");
    // Malformed: a file whose contents are refused.
    let dup = temp_file("config-dup", b"card = 300x200\ncard = 100x100\n");
    let err = refusal(&[("CRABCROP_PRESETS_FILE", dup.as_str())]);
    assert!(matches!(
        err,
        ConfigError::Malformed {
            var: "CRABCROP_PRESETS_FILE",
            ..
        }
    ));
    assert_eq!(err.to_string(), "duplicate preset name: card");
    assert!(std::error::Error::source(&err).is_none());
}

#[test]
fn from_env_reports_the_first_bad_setting_in_boot_order() {
    // One bad value per setting, in the order boot reads
    // them. Of any two, the one read first is reported,
    // exactly as when it is bad alone: the order of the
    // checks is part of the boot contract (issue #477).
    let missing = missing_path("order");
    let ordered: Vec<(&str, &str)> = vec![
        ("CRABCROP_SECRET_KEY_ID", "bad id!"),
        ("CRABCROP_ALLOW_UNSAFE", "ture"),
        ("PORT", "abc"),
        ("CRABCROP_SSIM_TARGET", "101"),
        ("CRABCROP_PROCESS_TIMEOUT_SECS", "0"),
        ("CRABCROP_SLOW_LANE_PERMITS", "65"),
        ("CRABCROP_SOURCE_MAX_MB", "0"),
        ("CRABCROP_SOURCE_TIMEOUT_SECS", "601"),
        ("CRABCROP_SHUTDOWN_GRACE_SECS", "x"),
        ("CRABCROP_SOURCE_BUFFER_MAX_MB", "1"),
        ("CRABCROP_AVIF_SPEED", "11"),
        ("CRABCROP_JXL_SPEED", "turbo"),
        ("CRABCROP_WIDTH_STEP", "0"),
        ("CRABCROP_CACHE_MAX_MB", "x"),
        ("CRABCROP_BLOCK_PRIVATE_SOURCES", "2"),
        ("CRABCROP_ALLOW_ANY_SOURCE", ""),
        ("CRABCROP_S3_ENDPOINT", "ftp://x"),
        ("AWS_ACCESS_KEY_ID", "AKID"),
        ("CRABCROP_GCS_ENDPOINT", "not a url"),
        ("CRABCROP_GCS_SECRET_KEY", "gcs-secret"),
        ("CRABCROP_FILE_ROOT", missing.as_str()),
        ("CRABCROP_UPLOAD_BUCKET", "unlisted"),
        ("CRABCROP_ONLY_PRESETS", "1"),
        ("CRABCROP_SOURCE_ROOTS_FILE", missing.as_str()),
        ("CRABCROP_DEFAULT_OPS", "garbage!!"),
        ("CRABCROP_METADATA", "keep"),
        ("CRABCROP_BRAND_COLOR", "white"),
        ("CRABCROP_SEGMENT_ENCODER_MODEL", "/m/e.onnx"),
        ("CRABCROP_CLIP_IMAGE_MODEL", "/m/clip.onnx"),
        ("CRABCROP_MODERATION_LABELS", "/m/labels.txt"),
        ("CRABCROP_AESTHETIC_HEAD", "/m/head.bin"),
        ("CRABCROP_OCR_DICT", "/m/dict.txt"),
        ("CRABCROP_DETECTORS_FILE", missing.as_str()),
        ("CRABCROP_VIDEO_TIMEOUT_SECS", "601"),
        ("CRABCROP_VIDEO_MAX_MB", "x"),
        ("CRABCROP_PDF_TIMEOUT_SECS", "0"),
        ("CRABCROP_C2PA_TSA_URL", "http://tsa.example.com"),
        ("CRABCROP_META_MODE", "enabled"),
    ];
    let alone: Vec<String> = ordered
        .iter()
        .map(|pair| refusal(&[*pair]).to_string())
        .collect();
    for (i, first) in ordered.iter().enumerate() {
        for later in &ordered[i + 1..] {
            assert_eq!(
                refusal(&[*first, *later]).to_string(),
                alone[i],
                "{first:?} with {later:?}"
            );
        }
    }
    // The setting a range depends on is read first.
    let err = refusal(&[
        ("CRABCROP_RESULT_STORAGE_DIR", "/var/cache/crabcrop"),
        ("CRABCROP_RESULT_STORAGE_MAX_MB", "x"),
        ("CRABCROP_DETECT_MODEL", "/m/det.onnx"),
        ("CRABCROP_DETECT_THRESHOLD", "x"),
    ]);
    assert!(
        err.to_string().contains("CRABCROP_RESULT_STORAGE_MAX_MB"),
        "{err}"
    );
}

#[test]
fn settings_read_only_alongside_another_are_ignored_alone() {
    // A bound whose setting is unset is never read, as
    // before: none of these refuses boot.
    for pair in [
        ("CRABCROP_RESULT_STORAGE_MAX_MB", "x"),
        ("CRABCROP_UPLOAD_MAX_MB", "x"),
        ("CRABCROP_UPLOAD_ALLOW_OVERWRITE", "maybe"),
        ("CRABCROP_CLIP_THRESHOLD", "x"),
        ("CRABCROP_DETECT_THRESHOLD", "x"),
        ("CRABCROP_SECRET_PREVIOUS_KEY_ID", "bad id!"),
        ("CRABCROP_OCR_LANGUAGE", "en"),
    ] {
        config_of(&[pair]).unwrap_or_else(|e| panic!("{pair:?}: {e}"));
    }
}

/// Every spelling of `secret` a rendering could carry: the
/// text, and the decimal byte list a derived `Debug` prints
/// for a `Vec<u8>`.
fn assert_no_secret(what: &str, rendered: &str, secret: &str) {
    assert!(
        !rendered.contains(secret),
        "{what} leaks {secret:?}: {rendered}"
    );
    let compact: String = rendered.split_whitespace().collect();
    let bytes: String = format!("{:?}", secret.as_bytes())
        .split_whitespace()
        .collect();
    let bytes = bytes.trim_start_matches('[').trim_end_matches(']');
    assert!(
        !compact.contains(bytes),
        "{what} leaks the bytes of {secret:?}"
    );
}

#[test]
fn config_errors_never_carry_a_secret() {
    // Issue #455's rule for boot errors: a refused secret is
    // named with its length, never its value, in the
    // message, in `{:?}`, and in `{:#?}`.
    for (var, secret, pairs) in [
        (
            "CRABCROP_SECRET",
            "hunter2-short",
            vec![("CRABCROP_SECRET", "hunter2-short")],
        ),
        (
            "CRABCROP_SECRET_PREVIOUS",
            "hunter2-prev",
            vec![
                ("CRABCROP_SECRET", SECRET),
                ("CRABCROP_SECRET_PREVIOUS", "hunter2-prev"),
            ],
        ),
    ] {
        let err = Config::from_env(env_of(&pairs)).map(|_| ()).unwrap_err();
        match &err {
            ConfigError::Invalid { var: v, value, .. } => {
                assert_eq!(*v, var);
                assert_eq!(*value, ConfigValue::Secret);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            err.to_string(),
            format!("{var} must be at least 16 bytes, got {}", secret.len())
        );
        for rendered in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
            assert_no_secret(var, &rendered, secret);
        }
        assert!(format!("{err:?}").contains("[redacted]"), "{err:?}");
    }
    // A half-set credential pair names the variables only.
    for pair in [
        ("AWS_SECRET_ACCESS_KEY", "aws-secret-wJalrXUtnFEMI"),
        ("CRABCROP_GCS_SECRET_KEY", "gcs-secret-bGoaV7gyqDX"),
    ] {
        let err = refusal(&[pair]);
        assert!(matches!(err, ConfigError::Conflict { .. }), "{err:?}");
        for rendered in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
            assert_no_secret(pair.0, &rendered, pair.1);
        }
    }
    // Any refusal of a secret variable keeps no value, so a
    // future check cannot leak one: not in the value, and not
    // in a message that quotes it, as the generic parsers'
    // messages do, quoted (`{:?}`) or bare.
    let value = "the-secret-value";
    for var in SECRET_VARIABLES {
        let refusals = [
            ConfigError::invalid(var, value, format!("{var} is refused")),
            ConfigError::invalid(var, value, format!("invalid {var}: {value:?}")),
            ConfigError::invalid(var, value, format!("{var} {value} is refused")),
            parse_env_value::<u16>(var, Some(value.to_string()), 0).unwrap_err(),
            parse_bool_value(var, Some(value.to_string()), false).unwrap_err(),
        ];
        for err in refusals {
            assert!(
                matches!(
                    &err,
                    ConfigError::Invalid {
                        value: ConfigValue::Secret,
                        ..
                    }
                ),
                "{var}: {err:?}"
            );
            for rendered in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
                assert_no_secret(var, &rendered, value);
            }
        }
    }
    assert_eq!(
        parse_env_value::<u16>("CRABCROP_METRICS_TOKEN", Some(value.to_string()), 0)
            .unwrap_err()
            .to_string(),
        "invalid CRABCROP_METRICS_TOKEN: [redacted]"
    );
    // A store endpoint is not a secret, but it can carry a
    // user and password: a refused one that does is kept only
    // as set, and neither message quotes it.
    for var in ["CRABCROP_S3_ENDPOINT", "CRABCROP_GCS_ENDPOINT"] {
        for endpoint in [
            "ftp://AKIAEXAMPLE:wJalrXUtnFEMIK7MDENG@minio.internal",
            "0AKIAEXAMPLE:wJalrXUtnFEMIK7MDENG@minio.internal",
        ] {
            let err = refusal(&[(var, endpoint)]);
            assert!(
                matches!(
                    &err,
                    ConfigError::Invalid {
                        var: v,
                        value: ConfigValue::Secret,
                        ..
                    } if *v == var
                ),
                "{err:?}"
            );
            for rendered in [err.to_string(), format!("{err:?}"), format!("{err:#?}")] {
                assert_no_secret(var, &rendered, "wJalrXUtnFEMIK7MDENG");
                assert_no_secret(var, &rendered, "AKIAEXAMPLE");
            }
        }
        // One without an `@` is kept for the caller.
        let err = refusal(&[(var, "ftp://minio.internal")]);
        assert!(
            format!("{err:?}").contains("\"ftp://minio.internal\""),
            "{err:?}"
        );
    }
    // A value that is not a secret is kept for the caller.
    let err = ConfigError::invalid("CRABCROP_WIDTH_STEP", "0", "refused");
    assert!(format!("{err:?}").contains("\"0\""), "{err:?}");
}

#[test]
fn config_debug_never_prints_a_secret() {
    let secrets = [
        ("CRABCROP_SECRET", "current-signing-key-0123456789"),
        (
            "CRABCROP_SECRET_PREVIOUS",
            "previous-signing-key-9876543210",
        ),
        ("AWS_SECRET_ACCESS_KEY", "aws-secret-key-wJalrXUtnFEMI"),
        ("AWS_SESSION_TOKEN", "aws-session-token-FwoGZXIvYXdz"),
        ("CRABCROP_GCS_SECRET_KEY", "gcs-hmac-secret-bGoa-V7g-yqDX"),
        ("CRABCROP_METRICS_TOKEN", "metrics-bearer-token-5f1c0e9a"),
    ];
    let mut pairs = secrets.to_vec();
    // An upload bucket puts an S3 store, with the AWS
    // credentials and the entry key, inside `Config`.
    pairs.extend([
        ("AWS_ACCESS_KEY_ID", "AKID"),
        ("CRABCROP_GCS_ACCESS_KEY", "GOOG1"),
        ("CRABCROP_ALLOWED_BUCKETS", "photos"),
        ("CRABCROP_UPLOAD_BUCKET", "photos"),
    ]);
    let config = Config::from_env(env_of(&pairs)).unwrap();
    assert!(config.upload.is_some());
    for rendered in [format!("{config:?}"), format!("{config:#?}")] {
        for (var, secret) in secrets {
            assert_no_secret(var, &rendered, secret);
        }
        assert!(rendered.contains("[redacted]"), "{rendered}");
    }
}
