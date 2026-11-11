use super::*;

#[test]
fn parses_a_keyed_signature() {
    let (sig, kid, rest) = parse_signature("/k2:a1b2c3d4/300x200/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Hmac("a1b2c3d4".to_string()));
    assert_eq!(kid.as_deref(), Some("k2"));
    assert_eq!(rest, "300x200/example.com/img.jpg");
}

#[test]
fn rejects_malformed_key_ids() {
    for path in [
        "/:a1b2c3d4/300x200/x.jpg",
        "/k ey:a1b2c3d4/300x200/x.jpg",
        "/k2:/300x200/x.jpg",
        "/k2:nothex!/300x200/x.jpg",
        "/k2:unsafe/300x200/x.jpg",
    ] {
        assert!(
            matches!(parse_signature(path), Err(ParseError::MalformedSignature)),
            "{path}"
        );
    }
    let long = format!("/{}/a1b2/300x200/x.jpg", "k".repeat(32));
    assert!(parse_signature(&long.replace("/a1b2/", ":a1b2/")).is_ok());
}

#[test]
fn keyring_verifies_across_a_rotation_window() {
    let ring = Keyring {
        current_id: "k-new".to_string(),
        current: b"new-secret-value".to_vec(),
        previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
    };
    let remainder = "300x200/example.com/img.jpg";
    let new_sig = Signature::Hmac(crabcrop_url::sign(b"new-secret-value", remainder).unwrap());
    let old_sig = Signature::Hmac(crabcrop_url::sign(b"old-secret-value", remainder).unwrap());
    // Keyed URLs consult exactly the named key.
    assert!(ring.verify(Some("k-new"), &new_sig, remainder, false));
    assert!(ring.verify(Some("k-old"), &old_sig, remainder, false));
    assert!(!ring.verify(Some("k-new"), &old_sig, remainder, false));
    assert!(!ring.verify(Some("k-old"), &new_sig, remainder, false));
    // Unkeyed URLs (everything minted before key IDs)
    // consult the whole ring.
    assert!(ring.verify(None, &new_sig, remainder, false));
    assert!(ring.verify(None, &old_sig, remainder, false));
    // An unknown key ID fails like any bad signature.
    assert!(!ring.verify(Some("k-who"), &new_sig, remainder, false));
    // Unsafe still follows the server flag only.
    assert!(ring.verify(None, &Signature::Unsafe, remainder, true));
    assert!(!ring.verify(Some("k-new"), &Signature::Unsafe, remainder, false));
}

#[test]
fn keyring_purge_verification_keeps_domain_separation() {
    let ring = Keyring {
        current_id: "k-new".to_string(),
        current: b"new-secret-value".to_vec(),
        previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
    };
    let target = "300x200/example.com/img.jpg";
    let old_purge = Signature::Hmac(crabcrop_url::purge_signature(b"old-secret-value", target));
    assert!(ring.verify_purge(Some("k-old"), &old_purge, target));
    assert!(ring.verify_purge(None, &old_purge, target));
    assert!(!ring.verify_purge(Some("k-new"), &old_purge, target));
    // An image signature under the old key is not a purge
    // signature under any routing.
    let image_sig = Signature::Hmac(crabcrop_url::sign(b"old-secret-value", target).unwrap());
    assert!(!ring.verify_purge(Some("k-old"), &image_sig, target));
}

#[test]
fn parses_unsafe_signature() {
    let (sig, _kid, rest) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Unsafe);
    assert_eq!(rest, "300x200/example.com/img.jpg");
}

#[test]
fn parses_hmac_signature() {
    let (sig, _kid, rest) = parse_signature("/a1b2c3d4/300x200/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Hmac("a1b2c3d4".to_string()));
    assert_eq!(rest, "300x200/example.com/img.jpg");
}

#[test]
fn signatures_have_one_canonical_spelling() {
    // Uppercase and non-hex spellings are parse errors
    // (audit PL-7); builders only ever emit lowercase hex.
    assert!(parse_signature("/A1B2C3D4/300x200/example.com/img.jpg").is_err());
    assert!(parse_signature("/not-a-sig/300x200/example.com/img.jpg").is_err());
    assert!(parse_signature("/a1b2c3d4/300x200/example.com/img.jpg").is_ok());
}

#[test]
fn signature_is_case_sensitive_but_unsafe_is_lowercase_only() {
    // "Unsafe" is not the magic word, and under the
    // canonical-spelling rule (PL-7) it is not lowercase
    // hex either: a parse error, not an HMAC signature.
    assert!(parse_signature("/Unsafe/300x200/example.com/img.jpg").is_err());
    let (sig, _kid, _) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Unsafe);
}

#[test]
fn works_without_leading_slash() {
    let (sig, _kid, rest) = parse_signature("unsafe/300x200/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Unsafe);
    assert_eq!(rest, "300x200/example.com/img.jpg");
}

#[test]
fn rejects_empty_paths() {
    for path in ["", "/"] {
        assert_eq!(
            parse_signature(path),
            Err(ParseError::EmptyPath),
            "path: {path:?}"
        );
    }
}

#[test]
fn rejects_lone_segment() {
    // A signature with nothing after it is not a valid image request.
    for path in ["/unsafe", "unsafe", "/a1b2c3"] {
        assert_eq!(
            parse_signature(path),
            Err(ParseError::MissingSignature),
            "path: {path:?}"
        );
    }
}

#[test]
fn rejects_empty_first_segment() {
    assert_eq!(
        parse_signature("//300x200/x.jpg"),
        Err(ParseError::EmptyPath)
    );
}

#[test]
fn keeps_image_url_with_slashes_intact() {
    let (sig, _kid, rest) =
        parse_signature("/unsafe/filters:grayscale()/https://example.com/a/b/c.jpg").unwrap();
    assert_eq!(sig, Signature::Unsafe);
    assert_eq!(rest, "filters:grayscale()/https://example.com/a/b/c.jpg");
}

#[test]
fn canonical_key_form_strips_expiry_segments() {
    let src = "https://example.com/a.jpg";
    assert_eq!(
        canonical_key_form(&format!("exp:1893456000/300x200/{src}")),
        format!("300x200/{src}")
    );
    assert_eq!(
        canonical_key_form(&format!("300x200/exp:1893456000/{src}")),
        format!("300x200/{src}")
    );
    assert_eq!(canonical_key_form(&format!("exp:1893456000/{src}")), src);
    assert_eq!(
        canonical_key_form(&format!("300x200/{src}")),
        format!("300x200/{src}")
    );
    // An exp: segment inside the source is URL text, kept.
    let tricky = "https://example.com/exp:123/a.jpg";
    assert_eq!(canonical_key_form(tricky), tricky);
    // maxage: is per-request metadata too: stripped wherever
    // it sits among the operations, kept inside the source.
    assert_eq!(
        canonical_key_form(&format!("maxage:60/300x200/{src}")),
        format!("300x200/{src}")
    );
    assert_eq!(
        canonical_key_form(&format!("300x200/maxage:60/{src}")),
        format!("300x200/{src}")
    );
    assert_eq!(canonical_key_form(&format!("maxage:60/{src}")), src);
    let tricky = "https://example.com/maxage:60/a.jpg";
    assert_eq!(canonical_key_form(tricky), tricky);
}

#[test]
fn canonical_key_form_passes_unparseable_remainders_through() {
    // A remainder that does not parse keys as itself,
    // byte for byte, and stays a fixed point: no source
    // boundary is known, so no segment is stripped, not
    // even a well-formed `exp:` or `maxage:` ahead of the
    // failing one. Purge targets are caller text, so the
    // set is hostile: malformed metadata, a bad operation
    // argument, a missing source, an empty remainder, a
    // long run of segments.
    let long = format!("{}filters:blur(x)/a.jpg", "exp:1/".repeat(10_000));
    let cases = [
        "".to_string(),
        "/".to_string(),
        "exp:/300x200/https://example.com/a.jpg".to_string(),
        "exp:12x/300x200/https://example.com/a.jpg".to_string(),
        "maxage:99999999999/https://example.com/a.jpg".to_string(),
        "exp:1893456000/maxage:60/maxage:/https://example.com/a.jpg".to_string(),
        "exp:1893456000/300x200/".to_string(),
        "maxage:60/300x200/filters:blur(nope)/https://example.com/a.jpg".to_string(),
        "exp:99999999999999999999999/https://example.com/a.jpg".to_string(),
        long,
    ];
    for remainder in &cases {
        assert!(
            parse_operations_with_expiry(remainder).is_err(),
            "the case must not parse: {remainder:.80}"
        );
        let key = canonical_key_form(remainder);
        assert_eq!(&key, remainder, "an unparseable remainder keys as itself");
        assert_eq!(canonical_key_form(&key), key, "the key is a fixed point");
    }
}

#[test]
fn parse_error_codes_are_the_frozen_v2_set() {
    let cases: Vec<(ParseError, &str)> = vec![
        (ParseError::EmptyPath, "empty_path"),
        (ParseError::MissingSignature, "missing_signature"),
        (ParseError::MalformedSignature, "malformed_signature"),
        (ParseError::InvalidSize("x".into()), "invalid_size"),
        (ParseError::InvalidCrop("x".into()), "invalid_crop"),
        (ParseError::InvalidCropRect("x".into()), "invalid_crop_rect"),
        (
            ParseError::InvalidFocalRect("x".into()),
            "invalid_focal_rect",
        ),
        (
            ParseError::InvalidFocalPoint("x".into()),
            "invalid_focal_point",
        ),
        (ParseError::InvalidPad("x".into()), "invalid_pad"),
        (ParseError::InvalidExpiry("x".into()), "invalid_expiry"),
        (ParseError::InvalidMaxAge("x".into()), "invalid_maxage"),
        (ParseError::InvalidDebug("x".into()), "invalid_debug"),
        (ParseError::UnknownPreset("x".into()), "unknown_preset"),
        (ParseError::InvalidPreset("x".into()), "invalid_preset"),
        (ParseError::PresetsOnly("x".into()), "presets_only"),
        (ParseError::InvalidFilters("x".into()), "invalid_filters"),
        (ParseError::InvalidCurve("x".into()), "invalid_curve"),
        (
            ParseError::InvalidConvolution("x".into()),
            "invalid_convolution",
        ),
        (ParseError::InvalidNoise("x".into()), "invalid_noise"),
        (ParseError::InvalidFrame("x".into()), "invalid_frame"),
        (ParseError::InvalidPage("x".into()), "invalid_page"),
        (ParseError::MissingSourceUrl, "missing_source_url"),
        // The engine's code, kept when the cap check moved
        // to parse time (issue #469).
        (ParseError::TooManyOperations(101), "unsupported_operation"),
    ];
    let mut seen = std::collections::HashSet::new();
    for (err, expected) in &cases {
        assert_eq!(err.code(), *expected);
        assert!(seen.insert(err.code()), "codes must be unique");
    }
    assert_eq!(seen.len(), cases.len());
    // The one shared code: an expansion past its length
    // cap is refused as the request over the operation cap
    // is, so the set stays frozen.
    assert_eq!(
        ParseError::ExpansionTooLong(MAX_EXPANDED_LEN + 1).code(),
        "unsupported_operation"
    );
    // So is a request over the watermark URL cap (issue #613).
    assert_eq!(
        ParseError::TooManyWatermarks(MAX_WATERMARK_URLS + 1).code(),
        "unsupported_operation"
    );
}

#[test]
fn canonical_key_collapses_spelling_variants() {
    let src = "https://example.com/a.jpg";
    let pairs = [
        // Format alias.
        ("filters:format(jpg)", "filters:format(jpeg)"),
        // Leading zeros.
        ("0300x0200", "300x200"),
        // Omitted axis vs explicit 0.
        ("300x", "300x0"),
        // Hex case in colors.
        ("filters:fill(FFAA00)", "filters:fill(ffaa00)"),
        // Split filter segments merge.
        (
            "filters:grayscale/filters:blur(2)",
            "filters:grayscale:blur(2)",
        ),
        // The default crop alignment is no alignment.
        ("crop:300x200:center", "crop:300x200"),
        // Bare trim is trim(10).
        ("filters:trim", "filters:trim(10)"),
        // The focal point zoom defaults to 1 and is
        // spelled out in the canonical form.
        ("focalpoint:0.75x0.4", "focalpoint:0.75x0.4:1"),
        // One radius is four.
        ("filters:radius(10)", "filters:radius(10,10,10,10)"),
        // Float spellings.
        ("filters:contrast(1.50)", "filters:contrast(1.5)"),
        ("filters:gamma(2.20)", "filters:gamma(2.2)"),
        ("filters:exposure(1.50)", "filters:exposure(1.5)"),
        ("filters:exposure(-2.0)", "filters:exposure(-2)"),
        (
            "filters:highlights_shadows(30.0,-20.50)",
            "filters:highlights_shadows(30,-20.5)",
        ),
        ("filters:vibrance(40.0)", "filters:vibrance(40)"),
        // Six-digit text colors gain an opaque alpha.
        (
            "filters:text(Hi,20,ffaa00,center)",
            "filters:text(Hi,20,ffaa00ff,center)",
        ),
    ];
    for (a, b) in pairs {
        assert_eq!(
            canonical_key_form(&format!("{a}/{src}")),
            canonical_key_form(&format!("{b}/{src}")),
            "{a} and {b} must share a key"
        );
    }
    // Expiry never enters the key.
    assert_eq!(
        canonical_key_form(&format!("exp:1893456000/300x200/{src}")),
        canonical_key_form(&format!("300x200/{src}"))
    );
    // Neither does the cache lifetime: two URLs that differ
    // only in maxage share one cache entry.
    assert_eq!(
        canonical_key_form(&format!("maxage:60/300x200/{src}")),
        canonical_key_form(&format!("maxage:31536000/300x200/{src}"))
    );
    assert_eq!(
        canonical_key_form(&format!("maxage:60/300x200/{src}")),
        canonical_key_form(&format!("300x200/{src}"))
    );
    // Order is meaning, not spelling: it stays in the key.
    assert_ne!(
        canonical_key_form(&format!("filters:blur(2):grayscale/{src}")),
        canonical_key_form(&format!("filters:grayscale:blur(2)/{src}"))
    );
    // Pixels(0) is an exact size, not proportional.
    assert_ne!(
        canonical_key_form(&format!("00x200/{src}")),
        canonical_key_form(&format!("0x200/{src}"))
    );
}

#[test]
fn canonical_form_is_a_fixed_point_and_round_trips() {
    let src = "https://example.com/a.jpg";
    let corpus = [
        "300x200",
        "-300xorig",
        "crop:100x100:smart",
        "crop:100x100:bottom-right",
        "crop-rect:1x2:30x40",
        "focal:1x2:30x40",
        "focal:1x2:30x40/crop:100x100:smart",
        "debug:focal",
        "debug:focal/crop:100x100:smart",
        "maxage:60/300x200",
        "fit-in:300x200",
        "fit-in:300x200:adaptive",
        "pad:300x200",
        "pad:300x200:blur",
        "pad:300x200:A0B0C0",
        "filters:grayscale:invert:blur(1.5):brighten(-20):contrast(1.25)",
        "filters:huerotate(90):sharpen(0.5):saturation(2):rgb(1,-2,3):equalize",
        "filters:rotate(-90):trim(25):format(avif):quality(auto):quality(80)",
        "filters:watermark(https%3A%2F%2Fcdn.example.com%2Flogo.png,10,-10,80)",
        "filters:overlay(https%3A%2F%2Fe.com%2Fo.png,0,0,50,multiply,25)",
        "filters:removebg:keep_profile:blurfaces:pixelatefaces:no_upscale",
        "filters:blurclass(person):pixelateclass(license%20plate):facepad(0.5)",
        "filters:upscale(2):dpr(1.5):max_bytes(50000)",
        "filters:text(Hello%2C%20world,24,FFAA00CC,bottom-right)",
        "filters:fill(00ff00):radius(1,2,3,4):border(3,102030):mask(ellipse)",
        "filters:sepia:duotone(101010,f0e0c0):monochrome(336699):pixelate(8)",
        "filters:curve(rgb,0,0,255,255):curve(b,32,10,224,245)",
        "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)",
        "filters:convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)",
        "filters:noise(0)",
        "filters:noise(40)",
        "filters:flip(h):blur_region(1,2,30,40):pixelate_region(5,6,70,80)",
        "300x200/filters:grayscale",
        "filters:grayscale/300x200/filters:blur(2)",
    ];
    for text in corpus {
        let remainder = format!("{text}/{src}");
        let canonical = canonical_key_form(&remainder);
        assert_eq!(
            canonical_key_form(&canonical),
            canonical,
            "canonical form of {text} is not a fixed point"
        );
        let (ops_before, _, _, _) = parse_operations_with_expiry(&remainder).unwrap();
        let (ops_after, _, _, _) = parse_operations_with_expiry(&canonical).unwrap();
        assert_eq!(ops_before, ops_after, "canonical form changed {text}");
    }
}

#[test]
fn purge_signatures_are_domain_separated() {
    let secret = b"test-secret";
    let target = "300x200/https://example.com/a.jpg";
    let purge_sig = purge_signature(secret, target);
    // The replay from audit P-1: an image signature over
    // "purge/<target>" must not verify as a purge signature.
    let image_sig = crabcrop_url::sign(secret, &format!("purge/{target}")).unwrap();
    assert_ne!(purge_sig, image_sig);
    assert!(verify_purge(secret, &Signature::Hmac(purge_sig), target));
    assert!(!verify_purge(secret, &Signature::Hmac(image_sig), target));
    // Purges never honor unsafe (Q3), in any mode.
    assert!(!verify_purge(secret, &Signature::Unsafe, target));
}

#[test]
fn upload_signatures_are_domain_separated() {
    let secret = b"test-secret";
    let name = "heroes/cat.jpg";
    let upload_sig = upload_signature(secret, name);
    assert_eq!(
        upload_sig,
        "81d87b588203fdad32f5431c0a893bf7ddad8188f5cca123a1b057a439b34b84"
    );
    assert!(verify_upload(
        secret,
        &Signature::Hmac(upload_sig.clone()),
        name
    ));
    // Neither an image signature nor a purge signature
    // over the same string verifies as an upload
    // signature, and the upload signature verifies in
    // neither other domain.
    let image_sig = crabcrop_url::sign(secret, name).unwrap();
    let purge_sig = purge_signature(secret, name);
    assert!(!verify_upload(
        secret,
        &Signature::Hmac(image_sig.clone()),
        name
    ));
    assert!(!verify_upload(secret, &Signature::Hmac(purge_sig), name));
    assert!(!verify_purge(
        secret,
        &Signature::Hmac(upload_sig.clone()),
        name
    ));
    assert!(!verify(
        secret,
        &Signature::Hmac(upload_sig.clone()),
        name,
        false
    ));
    // Uploads never honor unsafe, in any mode.
    assert!(!verify_upload(secret, &Signature::Unsafe, name));
    // A signature for one name authorizes no other.
    assert!(!verify_upload(
        secret,
        &Signature::Hmac(upload_sig),
        "heroes/dog.jpg"
    ));
}

#[test]
fn upload_remainders_split_an_optional_expiry() {
    // The plain form is the name alone.
    assert_eq!(
        split_upload_remainder("heroes/cat.jpg").unwrap(),
        (None, "heroes/cat.jpg")
    );
    assert_eq!(
        split_upload_remainder("cat.jpg").unwrap(),
        (None, "cat.jpg")
    );
    // The expiring form leads with the exp: segment.
    assert_eq!(
        split_upload_remainder("exp:1893456000/heroes/cat.jpg").unwrap(),
        (Some(1893456000), "heroes/cat.jpg")
    );
    // A segment that opens with exp: but does not parse
    // is a grammar error with the frozen code, never a
    // name (names cannot contain ':' anyway).
    for bad in [
        "exp:soon/cat.jpg",
        "exp:/cat.jpg",
        "exp:-5/cat.jpg",
        "exp:99999999999999999999999999/cat.jpg",
        "exp:1893456000",
    ] {
        let err = split_upload_remainder(bad).unwrap_err();
        assert_eq!(err.code(), "invalid_expiry", "{bad}");
    }
    // A name segment merely resembling the prefix
    // without the colon is an ordinary name.
    assert_eq!(
        split_upload_remainder("expedition/cat.jpg").unwrap(),
        (None, "expedition/cat.jpg")
    );
}

#[test]
fn expiring_upload_signatures_verify_over_the_full_remainder() {
    let secret = b"test-secret";
    let name = "heroes/cat.jpg";
    let exp = 1893456000u64;
    let sig = upload_signature_with_expiry(secret, name, exp);
    let remainder = format!("exp:{exp}/{name}");
    // Verification runs over the remainder exactly as
    // presented, expiry segment included.
    assert!(verify_upload(
        secret,
        &Signature::Hmac(sig.clone()),
        &remainder
    ));
    // The same token authorizes no other name and no
    // other expiry: both are inside the signed string.
    assert!(!verify_upload(
        secret,
        &Signature::Hmac(sig.clone()),
        "exp:1893456000/heroes/dog.jpg"
    ));
    assert!(!verify_upload(
        secret,
        &Signature::Hmac(sig.clone()),
        "exp:1893456001/heroes/cat.jpg"
    ));
    // Stripping the expiry to replay the token as a
    // plain one fails, and the plain token gains no
    // expiry form by presentation.
    assert!(!verify_upload(secret, &Signature::Hmac(sig), name));
    let plain = upload_signature(secret, name);
    assert!(!verify_upload(secret, &Signature::Hmac(plain), &remainder));
}

#[test]
fn sealed_segment_splitting() {
    assert_eq!(split_sealed_segment("s1.AAECAw"), Some((None, "s1.AAECAw")));
    assert_eq!(
        split_sealed_segment("k2:s1.AAECAw"),
        Some((Some("k2"), "s1.AAECAw"))
    );
    // Plain spellings never split as sealed: a hex
    // signature, `unsafe`, a keyed signature, and
    // near-miss markers.
    for plain in [
        "0123456789abcdef",
        "unsafe",
        "k2:0123456789abcdef",
        "s1",
        "s12.AAAA",
        "k ey:s1.AAAA",
        ":s1.AAAA",
    ] {
        assert_eq!(split_sealed_segment(plain), None, "{plain}");
    }
    // Marker-shaped imitations still present as sealed
    // (they earn the uniform refusal, not the grammar's
    // 400), while ordinary segments do not.
    for imitation in ["s2.AAAA", "s12.x", "k2:s2.AAAA", "s1.anything"] {
        assert!(looks_sealed(imitation), "{imitation}");
    }
    for plain in ["unsafe", "0123456789abcdef", "sx.AAAA", "s.AAAA", "k2:a1b2"] {
        assert!(!looks_sealed(plain), "{plain}");
    }
}

#[test]
fn keyring_opens_sealed_tokens_across_rotation() {
    let remainder = "300x200/https://example.com/photo.jpg";
    let ring = Keyring {
        current_id: "k-new".to_string(),
        current: b"new-secret-value".to_vec(),
        previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
    };
    // A token minted under the previous secret opens
    // during the rotation window: unkeyed (current is
    // tried, then previous) and keyed to the old ID.
    let old_token = crabcrop_url::seal(b"old-secret-value", remainder).unwrap();
    assert_eq!(
        ring.open_sealed(None, &old_token).as_deref(),
        Some(remainder)
    );
    assert_eq!(
        ring.open_sealed(Some("k-old"), &old_token).as_deref(),
        Some(remainder)
    );
    // Keyed routing is exact: naming the current ID does
    // not open a previous-key token, and an unknown ID
    // opens nothing.
    assert_eq!(ring.open_sealed(Some("k-new"), &old_token), None);
    assert_eq!(ring.open_sealed(Some("k-nope"), &old_token), None);
    // A token under a foreign secret opens under no
    // routing.
    let foreign = crabcrop_url::seal(b"someone-elses-secret", remainder).unwrap();
    assert_eq!(ring.open_sealed(None, &foreign), None);
    // Once the previous key leaves the ring, its tokens
    // stop opening: the window has closed.
    let rotated = Keyring {
        current_id: "k-new".to_string(),
        current: b"new-secret-value".to_vec(),
        previous: None,
    };
    assert_eq!(rotated.open_sealed(None, &old_token), None);
    assert_eq!(rotated.open_sealed(Some("k-old"), &old_token), None);
    // The current key's tokens open throughout.
    let new_token = crabcrop_url::seal(b"new-secret-value", remainder).unwrap();
    assert_eq!(
        rotated.open_sealed(None, &new_token).as_deref(),
        Some(remainder)
    );
    // A tampered token opens nowhere.
    let mut bytes = old_token.into_bytes();
    let last = bytes.len() - 1;
    bytes[last] = if bytes[last] == b'A' { b'B' } else { b'A' };
    let tampered = String::from_utf8(bytes).unwrap();
    assert_eq!(ring.open_sealed(None, &tampered), None);
}

#[test]
fn keyring_upload_verification_keeps_domain_separation() {
    let ring = Keyring {
        current_id: "k-new".to_string(),
        current: b"new-secret-value".to_vec(),
        previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
    };
    let name = "heroes/cat.jpg";
    let old_upload = Signature::Hmac(crabcrop_url::upload_signature(b"old-secret-value", name));
    assert!(ring.verify_upload(Some("k-old"), &old_upload, name));
    assert!(ring.verify_upload(None, &old_upload, name));
    assert!(!ring.verify_upload(Some("k-new"), &old_upload, name));
    // An image signature under the old key is not an
    // upload signature under any routing.
    let image_sig = Signature::Hmac(crabcrop_url::sign(b"old-secret-value", name).unwrap());
    assert!(!ring.verify_upload(Some("k-old"), &image_sig, name));
}

#[test]
fn dimensions_are_capped_at_parse_time() {
    assert!(parse_size("16383x100").is_ok());
    assert!(parse_size("16384x100").is_err());
    assert!(parse_size("100x16384").is_err());
    // The cap applies wherever a size parses.
    assert!(crate::url::parse_operations("fit-in:16384x10").is_err());
    assert!(crate::url::parse_operations("pad:20000x20000").is_err());
    assert!(crate::url::parse_operations("crop:16384x10").is_err());
}

#[test]
fn malformed_size_segments_are_parse_errors_not_sources() {
    // Issue #228: an over-cap or garbled size segment used
    // to fall out of the parse and be treated as the start
    // of a source URL, surfacing as a 502 fetch failure.
    for input in [
        "16400x100/example.com/img.jpg",
        "16384x100/example.com/img.jpg",
        "99999x100/example.com/img.jpg",
        "abcx100/example.com/img.jpg",
        "300xabc/example.com/img.jpg",
    ] {
        assert_eq!(
            parse_operations(input),
            Err(ParseError::InvalidSize(
                input.split('/').next().unwrap().to_string()
            )),
            "input: {input:?}"
        );
    }
    // Source segments that merely contain an `x` still
    // start the source: hosts, file names, and schemes are
    // never size attempts.
    for (input, source) in [
        ("example.com/img.jpg", "example.com/img.jpg"),
        ("300x200/example.com/img.jpg", "example.com/img.jpg"),
        ("300x200/100x200.jpg", "100x200.jpg"),
    ] {
        let (_, parsed_source) = parse_operations(input).unwrap();
        assert_eq!(parsed_source, source, "input: {input:?}");
    }
}

#[test]
fn parses_sizes() {
    use Dimension::{Original, Pixels, Proportional};

    let size = |width, height, flip_horizontal, flip_vertical| ImageSize {
        width,
        height,
        flip_horizontal,
        flip_vertical,
    };
    let plain = |width, height| size(width, height, false, false);

    let cases = [
        ("300x200", plain(Pixels(300), Pixels(200))),
        ("-300x200", size(Pixels(300), Pixels(200), true, false)),
        ("300x-200", size(Pixels(300), Pixels(200), false, true)),
        ("-300x-200", size(Pixels(300), Pixels(200), true, true)),
        ("300x0", plain(Pixels(300), Proportional)),
        ("300x", plain(Pixels(300), Proportional)),
        ("x200", plain(Proportional, Pixels(200))),
        ("0x200", plain(Proportional, Pixels(200))),
        ("0x0", plain(Proportional, Proportional)),
        ("origx100", plain(Original, Pixels(100))),
        ("100xorig", plain(Pixels(100), Original)),
        ("origxorig", plain(Original, Original)),
        ("-0x-0", size(Proportional, Proportional, true, true)),
    ];
    for (input, expected) in cases {
        assert_eq!(parse_size(input), Ok(expected), "input: {input:?}");
    }
}

#[test]
fn rejects_invalid_sizes() {
    for input in ["", "300", "300xx200", "abcx200", "300xabc", "12.5x100"] {
        assert_eq!(
            parse_size(input),
            Err(ParseError::InvalidSize(input.to_string())),
            "input: {input:?}"
        );
    }
}

#[test]
fn signature_then_size_compose() {
    // The two primitives chain: signature off the front, size next.
    let (sig, _kid, rest) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Unsafe);
    let (size_seg, image) = rest.split_once('/').unwrap();
    let size = parse_size(size_seg).unwrap();
    assert_eq!(size.width, Dimension::Pixels(300));
    assert_eq!(size.height, Dimension::Pixels(200));
    assert_eq!(image, "example.com/img.jpg");
}

/// HMAC-SHA256 with secret `Jefe` over `KNOWN_REMAINDER`.
/// Cross-checked against Python's hmac module and OpenSSL before
/// it was written into this test.
const KNOWN_HMAC: &str = "ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da9";
const KNOWN_REMAINDER: &str = "what do ye want for nothing?";

fn known_signature() -> Signature {
    Signature::Hmac(KNOWN_HMAC.to_string())
}

#[test]
fn verify_accepts_valid_hmac_sha256() {
    assert!(verify(b"Jefe", &known_signature(), KNOWN_REMAINDER, false));
}

#[test]
fn verify_rejects_tampered_remainder() {
    assert!(!verify(
        b"Jefe",
        &known_signature(),
        "what do ye want for something?",
        false
    ));
}

#[test]
fn verify_rejects_wrong_secret() {
    assert!(!verify(
        b"NotJefe",
        &known_signature(),
        KNOWN_REMAINDER,
        false
    ));
}

#[test]
fn verify_unsafe_honored_only_when_enabled() {
    assert!(verify(b"secret", &Signature::Unsafe, "anything", true));
    assert!(!verify(b"secret", &Signature::Unsafe, "anything", false));
}

#[test]
fn verify_rejects_malformed_signatures_without_panicking() {
    for bad in [
        "",
        "abc",      // odd length, not decodable as hex bytes
        "zz",       // not hex at all
        "ba5396e1", // valid hex, wrong length
        "ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da", // 63 chars, odd
    ] {
        let sig = Signature::Hmac(bad.to_string());
        assert!(
            !verify(b"Jefe", &sig, KNOWN_REMAINDER, false),
            "signature: {bad:?}"
        );
    }
}

#[test]
fn parsed_signature_verifies_against_remainder() {
    let path = format!("/{KNOWN_HMAC}/{KNOWN_REMAINDER}");
    let (sig, _kid, rest) = parse_signature(&path).unwrap();
    assert!(verify(b"Jefe", &sig, rest, false));
}

fn size_op(width: u32, height: u32) -> Operation {
    Operation::Size(ImageSize {
        width: Dimension::Pixels(width),
        height: Dimension::Pixels(height),
        flip_horizontal: false,
        flip_vertical: false,
    })
}

#[test]
fn parses_size_operation() {
    let (ops, url) = parse_operations("300x200/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![size_op(300, 200)]);
    assert_eq!(url, "example.com/img.jpg");
}

#[test]
fn parses_crop_with_default_center() {
    let (ops, _) = parse_operations("crop:300x200/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: Align::Center,
        }]
    );
}

#[test]
fn parses_crop_with_alignment() {
    let (ops, _) = parse_operations("crop:300x200:top/example.com/img.jpg").unwrap();
    assert!(matches!(
        ops[0],
        Operation::Crop {
            align: Align::Top,
            ..
        }
    ));
}

#[test]
fn parses_crop_with_corner_alignment() {
    for (spelling, want) in [
        ("top-left", Align::TopLeft),
        ("top-right", Align::TopRight),
        ("bottom-left", Align::BottomLeft),
        ("bottom-right", Align::BottomRight),
    ] {
        let path = format!("crop:300x200:{spelling}/example.com/img.jpg");
        let (ops, _) = parse_operations(&path).unwrap();
        assert!(
            matches!(
                &ops[0],
                Operation::Crop { align, .. } if align == &want
            ),
            "spelling: {spelling}"
        );
    }
}

#[test]
fn rejects_bad_crops() {
    for seg in [
        "crop:",
        "crop:300",
        "crop:300x200:sideways",
        "crop:300x200:center:extra",
        // Corner spellings are vertical-first, hyphenated:
        // near-misses do not parse.
        "crop:300x200:left-top",
        "crop:300x200:top_left",
        "crop:300x200:topleft",
        // The exact-box contract: pixels on both axes,
        // no flips (fit-in owns proportional boxes, and
        // a flip is filters:flip).
        "crop:300x0",
        "crop:0x200",
        "crop:origx100",
        "crop:300xorig",
        "crop:-300x200",
        "crop:300x-200",
        "crop:00x200",
    ] {
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidCrop(seg.to_string())),
            "segment: {seg:?}"
        );
    }
}

#[test]
fn parses_filters() {
    let (ops, _) =
        parse_operations("filters:grayscale():quality(80):format(webp)/example.com/img.jpg")
            .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::Grayscale,
            Filter::Quality(Quality::Value(80)),
            Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
        ])]
    );
}

#[test]
fn parses_shape_filters() {
    let (ops, _) =
        parse_operations("filters:radius(12):border(2,00ff00):mask(ellipse)/example.com/img.jpg")
            .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::Radius([12, 12, 12, 12]),
            Filter::Border(2, [0, 255, 0]),
            Filter::Mask(MaskShape::Ellipse),
        ])]
    );
    let (ops, _) =
        parse_operations("filters:radius(1,2,3,4):mask(circle)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::Radius([1, 2, 3, 4]),
            Filter::Mask(MaskShape::Circle),
        ])]
    );
}

#[test]
fn parses_finishing_filters() {
    let (ops, _) = parse_operations(
        "filters:sepia:duotone(000000,ffffff):monochrome(ff8000):pixelate(8):flip(h)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::Sepia,
            Filter::Duotone([0, 0, 0], [255, 255, 255]),
            Filter::Monochrome([255, 128, 0]),
            Filter::Pixelate(8),
            Filter::Flip(FlipDirection::Horizontal),
        ])]
    );
}

#[test]
fn parses_region_filters() {
    let (ops, _) = parse_operations(
        "filters:blur_region(1,2,30,40):pixelate_region(0,0,5,5)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::BlurRegion {
                x: 1,
                y: 2,
                w: 30,
                h: 40
            },
            Filter::PixelateRegion {
                x: 0,
                y: 0,
                w: 5,
                h: 5
            },
        ])]
    );
}

#[test]
fn parses_overlay() {
    let (ops, _) = parse_operations(
        "filters:overlay(https%3A%2F%2Fcdn.example.com%2Fov.png,10,-5,80,multiply)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Overlay {
            url: "https://cdn.example.com/ov.png".to_string(),
            x: 10,
            y: -5,
            alpha: 80,
            blend: BlendMode::Multiply,
            width: None,
        }])]
    );
    // Width form, and a comma inside the URL itself.
    let (ops, _) = parse_operations(
        "filters:overlay(https%3A%2F%2Fx%2Fa,b.png,0,0,100,screen,25)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Overlay {
            url: "https://x/a,b.png".to_string(),
            x: 0,
            y: 0,
            alpha: 100,
            blend: BlendMode::Screen,
            width: Some(25),
        }])]
    );
    // Overlay URLs join the prefetch set.
    let (ops, _) = parse_operations(
        "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,normal)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        RequestDirectives::analyze(&ops).watermark_urls,
        ["https://x/ov.png"]
    );
}

#[test]
fn numeric_filter_bounds_accept_the_endpoints() {
    // The AP-6 validation table: the endpoints are in,
    // one step past them is a parse error (pinned in
    // rejects_bad_filters).
    for seg in [
        "filters:brighten(255)",
        "filters:brighten(-255)",
        "filters:contrast(100)",
        "filters:contrast(-100)",
        "filters:gamma(10)",
        "filters:gamma(0.1)",
        "filters:exposure(10)",
        "filters:exposure(-10)",
        "filters:exposure(0)",
        "filters:highlights_shadows(100,100)",
        "filters:highlights_shadows(-100,-100)",
        "filters:vibrance(100)",
        "filters:vibrance(-100)",
        "filters:saturation(0)",
        "filters:saturation(255)",
        "filters:quality(1)",
        "filters:quality(100)",
    ] {
        assert!(
            parse_operations(&format!("{seg}/example.com/img.jpg")).is_ok(),
            "{seg}"
        );
    }
}

#[test]
fn parses_gamma_filter() {
    let (ops, _) = parse_operations("filters:gamma(2.2)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Gamma(2.2)])]);
}

#[test]
fn parses_exposure_filter() {
    let (ops, _) = parse_operations("filters:exposure(1.5)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Exposure(1.5)])]);
    // Zero stops is the identity and parses, like
    // contrast at 0; the endpoints of the stops range
    // parse too (one step past them is pinned in
    // rejects_bad_filters).
    let (ops, _) = parse_operations("filters:exposure(-2)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Exposure(-2.0)])]);
    assert!(parse_operations("filters:exposure(0)/example.com/img.jpg").is_ok());
}

#[test]
fn parses_highlights_shadows_filter() {
    let (ops, _) =
        parse_operations("filters:highlights_shadows(30,-20)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::HighlightsShadows(
            30.0, -20.0
        )])]
    );
    // Both at 0 are the identity and parse, like
    // contrast at 0.
    let (ops, _) = parse_operations("filters:highlights_shadows(0,0)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::HighlightsShadows(
            0.0, 0.0
        )])]
    );
}

#[test]
fn parses_vibrance_filter() {
    let (ops, _) = parse_operations("filters:vibrance(40)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Vibrance(40.0)])]);
    // Zero is the identity and parses, like contrast
    // at 0.
    let (ops, _) = parse_operations("filters:vibrance(0)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Vibrance(0.0)])]);
}

#[test]
fn parses_enhance_filter() {
    let (ops, _) = parse_operations("filters:enhance/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Enhance])]);
    // Empty parentheses spell the same filter, like
    // every no-argument filter.
    let (ops, _) = parse_operations("filters:enhance()/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Enhance])]);
}

#[test]
fn parses_palette_filter() {
    let (ops, _) = parse_operations("filters:palette(64)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Palette(64)])]);
    // The endpoints of the color budget parse.
    assert!(parse_operations("filters:palette(2)/example.com/img.jpg").is_ok());
    assert!(parse_operations("filters:palette(256)/example.com/img.jpg").is_ok());
    for seg in [
        "filters:palette(0)",
        "filters:palette(1)",
        "filters:palette(257)",
        "filters:palette(-8)",
        "filters:palette()",
        "filters:palette(64,64)",
    ] {
        assert!(
            parse_operations(&format!("{seg}/example.com/img.jpg")).is_err(),
            "{seg}"
        );
    }
}

#[test]
fn parses_border_filter() {
    // The ends of the range parse: 0 (a no-op, like
    // radius) and the dimension cap; one past the cap is
    // pinned in rejects_bad_filters.
    let (ops, _) = parse_operations("filters:border(0,FF0000)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Border(0, [255, 0, 0])])]
    );
    assert!(parse_operations("filters:border(16383,FF0000)/example.com/img.jpg").is_ok());
}

#[test]
fn rejects_bad_filters() {
    for seg in [
        "filters:",
        "filters:blur()",
        "filters:quality(200)",
        "filters:quality(0)",
        "filters:brighten(256)",
        "filters:brighten(-256)",
        "filters:contrast(101)",
        "filters:contrast(-100.5)",
        "filters:gamma(0)",
        "filters:gamma(-1)",
        "filters:gamma(10.5)",
        "filters:gamma(nan)",
        "filters:gamma(inf)",
        "filters:gamma()",
        "filters:exposure(10.5)",
        "filters:exposure(-10.5)",
        "filters:exposure(nan)",
        "filters:exposure(inf)",
        "filters:exposure()",
        "filters:highlights_shadows(101,0)",
        "filters:highlights_shadows(0,-101)",
        "filters:highlights_shadows(30)",
        "filters:highlights_shadows(1,2,3)",
        "filters:highlights_shadows(nan,0)",
        "filters:highlights_shadows()",
        "filters:vibrance(101)",
        "filters:vibrance(-100.5)",
        "filters:vibrance(nan)",
        "filters:vibrance(inf)",
        "filters:vibrance()",
        "filters:saturation(256)",
        "filters:saturation(-1)",
        "filters:quality()",
        "filters:quality(autoo)",
        "filters:grayscale(1)",
        "filters:enhance(1)",
        "filters:format(bmp)",
        "filters:quality(80",
        "filters:radius()",
        "filters:radius(1,2)",
        "filters:radius(1,2,3)",
        "filters:radius(16384)",
        "filters:radius(-1)",
        "filters:border(16384,FF0000)",
        "filters:border(2,FFF)",
        "filters:border(2)",
        "filters:border(2,FF0000,1)",
        "filters:mask(square)",
        "filters:mask()",
        "filters:sepia(1)",
        "filters:duotone(FF0000)",
        "filters:duotone(FF0000,FFF)",
        "filters:monochrome(FFF)",
        "filters:pixelate(0)",
        "filters:pixelate(16384)",
        "filters:flip(x)",
        "filters:flip()",
        "filters:blur_region(1,2,0,40)",
        "filters:blur_region(1,2,3)",
        "filters:blur_region(1,2,3,4,5)",
        "filters:blur_region(-1,0,10,10)",
        "filters:pixelate_region(0,0,10,16384)",
        "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,101,normal)",
        "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,darken)",
        "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,normal,0)",
        "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,normal,401)",
        "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100)",
        "filters:overlay(,0,0,100,normal)",
    ] {
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidFilters(seg.to_string())),
            "segment: {seg:?}"
        );
    }
}

#[test]
fn parses_curve_filter() {
    let (ops, _) = parse_operations(
        "filters:curve(rgb,0,0,255,255):curve(r,0,64,255,255)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::Curve(CurveChannel::Rgb, vec![(0, 0), (255, 255)]),
            Filter::Curve(CurveChannel::R, vec![(0, 64), (255, 255)]),
        ])]
    );
    // The point-count endpoints: 2 and 16 points parse.
    assert!(parse_operations("filters:curve(g,0,0,255,255)/example.com/img.jpg").is_ok());
    let sixteen: Vec<String> = (0..16).map(|i| format!("{},{}", i * 17, i * 17)).collect();
    assert!(
        parse_operations(&format!(
            "filters:curve(b,{})/example.com/img.jpg",
            sixteen.join(",")
        ))
        .is_ok()
    );
    // Leading zeros and a `+` sign are spelling, not
    // meaning: the canonical form prints the plain values.
    assert_eq!(
        canonical_key_form("filters:curve(rgb,007,0,+255,255)/example.com/img.jpg"),
        "filters:curve(rgb,7,0,255,255)/example.com/img.jpg"
    );
}

#[test]
fn rejects_bad_curves() {
    // Every curve failure is invalid_curve, not the
    // segment-wide invalid_filters.
    for seg in [
        // Unknown channel.
        "filters:curve(x,0,0,255,255)",
        "filters:curve(RGB,0,0,255,255)",
        "filters:curve(l,0,0,255,255)",
        // Fewer than 2 points, and an odd coordinate count.
        "filters:curve(rgb)",
        "filters:curve(rgb,0,0)",
        "filters:curve(rgb,0,0,255)",
        "filters:curve(rgb,0,0,255,255,128)",
        // Coordinates outside 0-255.
        "filters:curve(rgb,0,0,256,255)",
        "filters:curve(rgb,0,-1,255,255)",
        "filters:curve(rgb,0,0,255,1.5)",
        // x not strictly increasing.
        "filters:curve(rgb,0,0,0,255)",
        "filters:curve(rgb,128,0,64,255)",
        "filters:curve(rgb,0,0,128,10,128,20)",
    ] {
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidCurve(seg.to_string())),
            "segment: {seg:?}"
        );
    }
    // 17 points is one past the cap.
    let seventeen: Vec<String> = (0..17).map(|i| format!("{},{}", i * 15, i * 15)).collect();
    let seg = format!("filters:curve(rgb,{})", seventeen.join(","));
    assert_eq!(
        parse_operations(&format!("{seg}/example.com/img.jpg")),
        Err(ParseError::InvalidCurve(seg.clone()))
    );
    // A bare `curve` with no argument list is a plain
    // invalid filters segment, like every other filter.
    assert_eq!(
        parse_operations("filters:curve/example.com/img.jpg"),
        Err(ParseError::InvalidFilters("filters:curve".to_string()))
    );
}

#[test]
fn parses_convolution_filter() {
    let (ops, _) =
        parse_operations("filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)/example.com/img.jpg")
            .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Convolution(
            vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            false,
        )])]
    );
    // Decimal values (a gaussian kernel) and the flag.
    let (ops, _) = parse_operations(
        "filters:convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Convolution(
            vec![
                0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625
            ],
            true,
        )])]
    );
    // A 5x5 kernel parses: 25 values plus the flag.
    let taps: Vec<String> = (0..25)
        .map(|i| {
            if i == 12 {
                "1".to_string()
            } else {
                "0".to_string()
            }
        })
        .collect();
    assert!(
        parse_operations(&format!(
            "filters:convolution({},0)/example.com/img.jpg",
            taps.join(",")
        ))
        .is_ok()
    );
    // The bound itself is allowed, on either sign.
    assert!(
        parse_operations("filters:convolution(1024,-1024,0,0,1,0,0,0,0,0)/example.com/img.jpg")
            .is_ok()
    );
    // Spelling variants collapse in the canonical form:
    // trailing fraction zeros and a `.0` are the value,
    // not the spelling.
    assert_eq!(
        canonical_key_form("filters:convolution(0.50,-1.0,0,0,1,0,0,0,0,0)/example.com/img.jpg"),
        "filters:convolution(0.5,-1,0,0,1,0,0,0,0,0)/example.com/img.jpg"
    );
}

#[test]
fn rejects_bad_convolutions() {
    // Every convolution failure is invalid_convolution,
    // not the segment-wide invalid_filters.
    for seg in [
        // Wrong counts: 8 values, 10 values, 24 values,
        // 26 values (each plus the flag).
        "filters:convolution(0,0,0,0,1,0,0,0,0)",
        "filters:convolution(0,0,0,0,1,0,0,0,0,0,0)",
        "filters:convolution(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)",
        "filters:convolution(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)",
        // The flag is exactly 0 or 1.
        "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,2)",
        "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,-1)",
        "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,true)",
        "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,01)",
        // Values outside the strict decimal grammar: no
        // exponent, no leading dot, no plus, no bare dot.
        "filters:convolution(0,-1,0,-1,1e2,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,.5,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,+5,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,5.,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,x,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,,-1,0,-1,0,0)",
        // Values past the kernel bound.
        "filters:convolution(0,-1,0,-1,1025,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,-1024.5,-1,0,-1,0,0)",
        "filters:convolution(0,-1,0,-1,999999999999999999999,-1,0,-1,0,0)",
    ] {
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidConvolution(seg.to_string())),
            "segment: {seg:?}"
        );
    }
    // A bare `convolution` with no argument list is a
    // plain invalid filters segment, like every other
    // filter.
    assert_eq!(
        parse_operations("filters:convolution/example.com/img.jpg"),
        Err(ParseError::InvalidFilters(
            "filters:convolution".to_string()
        ))
    );
}

#[test]
fn parses_noise_filter() {
    let (ops, _) = parse_operations("filters:noise(40)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Noise(40)])]);
    // The ends of the range parse: 0 (a no-op) and 100
    // (full scale).
    assert!(parse_operations("filters:noise(0)/example.com/img.jpg").is_ok());
    assert!(parse_operations("filters:noise(100)/example.com/img.jpg").is_ok());
    // Leading zeros and a `+` sign are spelling, not
    // meaning: the canonical form prints the plain value.
    assert_eq!(
        canonical_key_form("filters:noise(007)/example.com/img.jpg"),
        "filters:noise(7)/example.com/img.jpg"
    );
    assert_eq!(
        canonical_key_form("filters:noise(+40)/example.com/img.jpg"),
        "filters:noise(40)/example.com/img.jpg"
    );
}

#[test]
fn rejects_bad_noise() {
    // Every noise failure is invalid_noise, not the
    // segment-wide invalid_filters.
    for seg in [
        // Past the 0..=100 range (256 does not even
        // fit the u8 the amount parses as).
        "filters:noise(101)",
        "filters:noise(256)",
        // Not an integer in the grammar.
        "filters:noise(-1)",
        "filters:noise(1.5)",
        "filters:noise(abc)",
        "filters:noise(1,2)",
    ] {
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidNoise(seg.to_string())),
            "segment: {seg:?}"
        );
    }
    // A bare `noise` with no argument list is a plain
    // invalid filters segment, like every other filter.
    assert_eq!(
        parse_operations("filters:noise/example.com/img.jpg"),
        Err(ParseError::InvalidFilters("filters:noise".to_string()))
    );
}

#[test]
fn parses_dpr_directive() {
    let (ops, _) = parse_operations("filters:dpr(2)/example.com/i.png").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Dpr(2.0)])]);
    assert_eq!(RequestDirectives::analyze(&ops).dpr, Some(2.0));
    for bad in [
        "filters:dpr(1.25)", // not one of the settled factors
        "filters:dpr(0)",
        "filters:dpr(4)",
        "filters:dpr(abc)",
        "filters:dpr", // the factor is required
    ] {
        let path = format!("{bad}/example.com/i.png");
        assert!(
            matches!(parse_operations(&path), Err(ParseError::InvalidFilters(_))),
            "input: {bad}"
        );
    }
}

#[test]
fn parses_text_with_encoded_unicode_and_commas() {
    let (ops, _) = parse_operations(
        "filters:text(Hello%2C%20World!%20%E2%9C%93,32,ff8800,bottom-right)/example.com/i.png",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Text {
            text: "Hello, World! ✓".to_string(),
            size: 32,
            color: [255, 136, 0, 255],
            anchor: Anchor::BottomRight,
        }])]
    );
    // An 8-digit color carries its own alpha.
    let (ops, _) =
        parse_operations("filters:text(X,10,ff880080,center)/example.com/i.png").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Text {
            text: "X".to_string(),
            size: 10,
            color: [255, 136, 0, 128],
            anchor: Anchor::Center,
        }])]
    );
}

#[test]
fn rejects_bad_text_params() {
    for bad in [
        "filters:text(X,0,ff8800,center)",    // size 0
        "filters:text(X,1001,ff8800,center)", // size too big
        "filters:text(X,10,ff88,center)",     // short color
        "filters:text(X,10,zzzzzz,center)",   // non-hex color
        "filters:text(X,10,ff8800,middle)",   // unknown anchor
        "filters:text(X,10,ff8800)",          // missing field
        "filters:text(,10,ff8800,center)",    // empty text
        "filters:text(%zz,10,ff8800,center)", // bad encoding
    ] {
        let path = format!("{bad}/example.com/i.png");
        assert!(
            matches!(parse_operations(&path), Err(ParseError::InvalidFilters(_))),
            "input: {bad}"
        );
    }
}

#[test]
fn text_labels_past_the_caps_are_refused() {
    // Issue #606: drawing costs about the glyph count times
    // the size squared, so a label holds at most 1000
    // characters, and its character count times its size
    // is at most 100000. At either cap the label parses;
    // one past answers `invalid_filters`.
    let parse = |label: &str, size: u32| {
        parse_operations(&format!(
            "filters:text({label},{size},ffffff,top-left)/example.com/i.png"
        ))
        .map(|(ops, _)| ops)
    };
    let accepted = [
        ("a".repeat(1000), 100, 1000),
        ("W".repeat(100), 1000, 100),
        // Characters count, not bytes: these 1000 are 2000
        // bytes, 6000 percent-encoded.
        ("%C3%A9".repeat(1000), 100, 1000),
    ];
    for (label, size, chars) in accepted {
        let ops = parse(&label, size).unwrap_or_else(|e| panic!("{size}: {e}"));
        let [Operation::Filters(filters)] = &ops[..] else {
            panic!("not one filters segment: {ops:?}");
        };
        let [Filter::Text { text, .. }] = &filters[..] else {
            panic!("not one text filter: {filters:?}");
        };
        assert_eq!(text.chars().count(), chars);
    }
    let refused = [
        ("a".repeat(1001), 1),
        ("a".repeat(1001), 12),
        ("W".repeat(101), 1000),
        ("W".repeat(1001), 100),
        ("%C3%A9".repeat(1001), 12),
        // The issue's label: 6000 glyphs at size 1000.
        ("W".repeat(6000), 1000),
    ];
    for (label, size) in refused {
        assert!(
            matches!(parse(&label, size), Err(ParseError::InvalidFilters(_))),
            "{} bytes at size {size}",
            label.len()
        );
    }
}

#[test]
fn parses_jxl_format() {
    let (ops, _) = parse_operations("filters:format(jxl)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
            ImageFormat::Jxl
        ))])]
    );
    // `removebg` parses, and takes no parameters.
    let (ops, _) = parse_operations("filters:removebg/example.com/img.png").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::RemoveBg])]);
    // `upscale` takes a factor of 2 or 4, nothing else.
    let (ops, _) = parse_operations("filters:upscale(2)/example.com/img.png").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Upscale(2)])]);
    let (ops, _) = parse_operations("filters:upscale(4)/example.com/img.png").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Upscale(4)])]);
    assert!(parse_operations("filters:upscale(3)/example.com/img.png").is_err());
    assert!(parse_operations("filters:upscale(0)/example.com/img.png").is_err());
    assert!(parse_operations("filters:upscale/example.com/img.png").is_err());
}

#[test]
fn parses_auto_quality() {
    let (ops, _) = parse_operations("filters:quality(auto)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Quality(Quality::Auto)])]
    );
}

#[test]
fn parses_format_options() {
    let (ops, _) =
        parse_operations("filters:format(jpeg,progressive,subsampling:444)/example.com/i.jpg")
            .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec {
            format: ImageFormat::Jpeg,
            options: FormatOptions {
                jpeg_progressive: true,
                jpeg_subsampling: Some(Subsampling::S444),
                ..FormatOptions::default()
            },
        })])]
    );
    let (ops, _) = parse_operations("filters:format(png,interlace)/example.com/i.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec {
            format: ImageFormat::Png,
            options: FormatOptions {
                png_interlace: true,
                ..FormatOptions::default()
            },
        })])]
    );
    let (ops, _) =
        parse_operations("filters:format(webp,method:6,sns:20)/example.com/i.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec {
            format: ImageFormat::WebP,
            options: FormatOptions {
                webp_method: Some(6),
                webp_sns: Some(20),
                ..FormatOptions::default()
            },
        })])]
    );
    // A bare format is the plain spec every pre-options
    // URL already means.
    let (ops, _) = parse_operations("filters:format(webp)/example.com/i.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
            ImageFormat::WebP
        ))])]
    );
}

#[test]
fn format_options_canonicalize_in_fixed_order() {
    for (given, canonical) in [
        (
            "filters:format(webp,sns:20,method:6)",
            "filters:format(webp,method:6,sns:20)",
        ),
        (
            "filters:format(jpeg,subsampling:422,progressive)",
            "filters:format(jpeg,progressive,subsampling:422)",
        ),
        (
            "filters:format(png,interlace)",
            "filters:format(png,interlace)",
        ),
        ("filters:format(jpeg)", "filters:format(jpeg)"),
    ] {
        let url = format!("{given}/example.com/i.jpg");
        let (ops, source) = parse_operations(&url).unwrap();
        assert_eq!(
            canonical_form(&ops, source),
            format!("{canonical}/example.com/i.jpg"),
            "{given}"
        );
    }
}

#[test]
fn format_options_reject_what_encoders_cannot_honour() {
    for bad in [
        // Options on formats whose encoders have none.
        "filters:format(avif,subsampling:420)",
        "filters:format(avif,progressive)",
        "filters:format(jxl,method:4)",
        "filters:format(gif,interlace)",
        "filters:format(mp4,method:4)",
        // Another format's options.
        "filters:format(jpeg,interlace)",
        "filters:format(jpeg,method:4)",
        "filters:format(png,progressive)",
        "filters:format(png,method:4)",
        "filters:format(webp,progressive)",
        "filters:format(webp,subsampling:420)",
        // Unknown, duplicated, or out-of-range options.
        "filters:format(jpeg,fast)",
        "filters:format(jpeg,progressive,progressive)",
        "filters:format(jpeg,subsampling:421)",
        "filters:format(jpeg,subsampling:)",
        "filters:format(webp,method:7)",
        "filters:format(webp,method:-1)",
        "filters:format(webp,method:99999999999999999999)",
        "filters:format(webp,method:4,method:5)",
        "filters:format(webp,sns:101)",
        "filters:format(webp,sns:4.5)",
        "filters:format(webp,)",
        "filters:format(,method:4)",
        "filters:format(webp,m\u{e9}thod:4)",
    ] {
        assert!(
            parse_operations(&format!("{bad}/example.com/i.jpg")).is_err(),
            "{bad} must not parse"
        );
    }
}

#[test]
fn parses_per_format_quality() {
    let (ops, _) = parse_operations("filters:quality(jpeg:80,webp:75)/example.com/i.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Quality(
            Quality::PerFormat(vec![
                (Some(ImageFormat::Jpeg), QualityValue::Value(80)),
                (Some(ImageFormat::WebP), QualityValue::Value(75)),
            ],)
        )])]
    );
    // Fallback plus entries, with an auto entry.
    let (ops, _) =
        parse_operations("filters:quality(70,jpeg:auto,avif:60)/example.com/i.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Quality(
            Quality::PerFormat(vec![
                (None, QualityValue::Value(70)),
                (Some(ImageFormat::Jpeg), QualityValue::Auto),
                (Some(ImageFormat::Avif), QualityValue::Value(60)),
            ],)
        )])]
    );
    // A single named entry is a list of one.
    let (ops, _) = parse_operations("filters:quality(webp:75)/example.com/i.jpg").unwrap();
    assert!(matches!(
        ops[0],
        Operation::Filters(ref f) if matches!(f[0], Filter::Quality(Quality::PerFormat(_)))
    ));
}

#[test]
fn per_format_quality_canonicalizes_sorted_with_fallback_first() {
    for (given, canonical) in [
        (
            "filters:quality(webp:75,jpeg:80)",
            "filters:quality(jpeg:80,webp:75)",
        ),
        (
            "filters:quality(jpeg:90,70,avif:60)",
            "filters:quality(70,avif:60,jpeg:90)",
        ),
        ("filters:quality(jpg:80)", "filters:quality(jpeg:80)"),
    ] {
        let url = format!("{given}/example.com/i.jpg");
        let (ops, source) = parse_operations(&url).unwrap();
        assert_eq!(
            canonical_form(&ops, source),
            format!("{canonical}/example.com/i.jpg"),
            "{given}"
        );
    }
}

#[test]
fn per_format_quality_rejects_bad_lists() {
    for bad in [
        "filters:quality(jpeg:80,jpeg:90)",
        "filters:quality(70,80)",
        "filters:quality(70,auto)",
        "filters:quality(unknown:80)",
        "filters:quality(mp4:80)",
        "filters:quality(webm:80)",
        "filters:quality(jpeg:0)",
        "filters:quality(jpeg:101)",
        "filters:quality(jpeg:)",
        "filters:quality(:80)",
        "filters:quality(jpeg:80,)",
        "filters:quality(,jpeg:80)",
        "filters:quality(jpeg:80,,webp:75)",
        "filters:quality(jpeg:80:90)",
        "filters:quality(jpeg:99999999999999999999)",
    ] {
        assert!(
            parse_operations(&format!("{bad}/example.com/i.jpg")).is_err(),
            "{bad} must not parse"
        );
    }
}

#[test]
fn per_format_quality_resolves_for_the_negotiated_format() {
    let q = Quality::PerFormat(vec![
        (None, QualityValue::Value(70)),
        (Some(ImageFormat::Jpeg), QualityValue::Value(90)),
        (Some(ImageFormat::WebP), QualityValue::Auto),
    ]);
    assert_eq!(
        q.resolved_for(ImageFormat::Jpeg),
        Some(ResolvedQuality::Value(90))
    );
    assert_eq!(
        q.resolved_for(ImageFormat::WebP),
        Some(ResolvedQuality::Auto)
    );
    // Unlisted formats fall back.
    assert_eq!(
        q.resolved_for(ImageFormat::Avif),
        Some(ResolvedQuality::Value(70))
    );
    // No entry and no fallback: no directive in force.
    let q = Quality::PerFormat(vec![(Some(ImageFormat::Jpeg), QualityValue::Value(90))]);
    assert_eq!(q.resolved_for(ImageFormat::WebP), None);
    // Plain forms resolve to themselves for every format.
    assert_eq!(
        Quality::Value(55).resolved_for(ImageFormat::Png),
        Some(ResolvedQuality::Value(55))
    );
    assert_eq!(
        Quality::Auto.resolved_for(ImageFormat::Gif),
        Some(ResolvedQuality::Auto)
    );
}

#[test]
fn parses_studio_defaults_and_arguments() {
    // Bare `studio(product)` takes every default.
    let (ops, _) = parse_operations("filters:studio(product)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
            mode: StudioMode::Product,
            bg: StudioBg::Default,
            shadow: StudioShadow::Soft,
            pad: 10,
        })])]
    );
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "filters:studio(product)/example.com/img.jpg"
    );
    // The full argument set, spelled in any order, parses
    // to the same spec and canonicalizes in grammar
    // order (bg, shadow, pad).
    let (ops, _) = parse_operations(
        "filters:studio(product,pad:25,shadow:strong,bg:f5f0e8)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
            mode: StudioMode::Product,
            bg: StudioBg::Color([0xf5, 0xf0, 0xe8]),
            shadow: StudioShadow::Strong,
            pad: 25,
        })])]
    );
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "filters:studio(product,bg:f5f0e8,shadow:strong,pad:25)/example.com/img.jpg"
    );
    // Every backdrop spelling.
    for (spelling, bg) in [
        ("white", StudioBg::White),
        ("brand", StudioBg::Brand),
        ("none", StudioBg::None),
    ] {
        let (ops, _) = parse_operations(&format!(
            "filters:studio(product,bg:{spelling})/example.com/img.jpg"
        ))
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
                bg,
                ..StudioSpec::default()
            })])]
        );
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            format!("filters:studio(product,bg:{spelling})/example.com/img.jpg")
        );
    }
    // Arguments spelled at their defaults collapse into
    // the bare form, so the two spellings share a cache
    // entry (the values ARE the defaults).
    let (ops, _) =
        parse_operations("filters:studio(product,shadow:soft,pad:10)/example.com/img.jpg").unwrap();
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "filters:studio(product)/example.com/img.jpg"
    );
    // But `bg:white` is not the default: with a brand
    // color configured they resolve differently, so
    // they key apart.
    assert_ne!(
        canonical_key_form("filters:studio(product)/example.com/img.jpg"),
        canonical_key_form("filters:studio(product,bg:white)/example.com/img.jpg")
    );
}

#[test]
fn parses_studio_scene_mode() {
    // Bare `studio(scene)` (issue #198): the scene
    // mode at the default pad, keyed apart from every
    // product spelling.
    let (ops, _) = parse_operations("filters:studio(scene)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
            mode: StudioMode::Scene,
            ..StudioSpec::default()
        })])]
    );
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "filters:studio(scene)/example.com/img.jpg"
    );
    assert_ne!(
        canonical_key_form("filters:studio(scene)/example.com/img.jpg"),
        canonical_key_form("filters:studio(product)/example.com/img.jpg")
    );
    // `pad` is the one argument scene takes, with the
    // product range and default: spelled at the
    // default it collapses into the bare form.
    let (ops, _) = parse_operations("filters:studio(scene,pad:25)/example.com/img.jpg").unwrap();
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "filters:studio(scene,pad:25)/example.com/img.jpg"
    );
    let (ops, _) = parse_operations("filters:studio(scene,pad:10)/example.com/img.jpg").unwrap();
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "filters:studio(scene)/example.com/img.jpg"
    );
    // A scene request takes the studio preflight and
    // the AI gate, like product.
    let d = RequestDirectives::analyze(&ops);
    assert!(d.studio);
    assert!(d.uses_ai());
    // The product-only arguments are bad arguments
    // under scene, an unknown mode is not a mode, and
    // the pad bounds hold in both modes.
    for bad in [
        "filters:studio(scene,bg:white)/example.com/img.jpg",
        "filters:studio(scene,shadow:soft)/example.com/img.jpg",
        "filters:studio(scene,pad:41)/example.com/img.jpg",
        "filters:studio(scene,pad:-1)/example.com/img.jpg",
        "filters:studio(product,pad:41)/example.com/img.jpg",
        "filters:studio(portrait)/example.com/img.jpg",
        "filters:studio()/example.com/img.jpg",
        "filters:studio(scene,pad:10,pad:20)/example.com/img.jpg",
    ] {
        assert!(parse_operations(bad).is_err(), "{bad}");
    }
}

#[test]
fn rejects_bad_studio_arguments() {
    for seg in [
        // The mode is required, and only `product`
        // and `scene` are modes.
        "filters:studio",
        "filters:studio()",
        "filters:studio(product,bg:blur)",
        "filters:studio(product,bg:fffff)",
        "filters:studio(product,bg:fffffff)",
        "filters:studio(product,shadow:hard)",
        "filters:studio(product,pad:41)",
        "filters:studio(product,pad:-1)",
        "filters:studio(product,pad:1.5)",
        "filters:studio(product,zoom:2)",
        // A key may not repeat.
        "filters:studio(product,pad:5,pad:6)",
        "filters:studio(product,bg:white,bg:none)",
    ] {
        let path = format!("{seg}/example.com/img.jpg");
        let result = parse_operations(&path);
        assert!(
            matches!(result, Err(ParseError::InvalidFilters(_))),
            "{seg} should fail as invalid filters, got {result:?}"
        );
    }
}

#[test]
fn studio_marks_the_request_as_ai() {
    let (ops, _) = parse_operations("filters:studio(product)/example.com/img.jpg").unwrap();
    let d = RequestDirectives::analyze(&ops);
    assert!(d.studio);
    assert!(
        d.uses_ai(),
        "studio runs the mask model, so it takes an AI permit"
    );
}

#[test]
fn remove_region_parses_like_the_other_region_filters() {
    // The grammar mirrors `blur_region`/`pixelate_region`
    // exactly (issue #176): four pixel values, w and h
    // at least 1, in a `filters:` list, canonicalized
    // back to the same spelling.
    let (ops, _) = parse_operations(
        "filters:blur_region(1,2,30,40):remove_region(5,6,70,80)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::BlurRegion {
                x: 1,
                y: 2,
                w: 30,
                h: 40
            },
            Filter::RemoveRegion {
                x: 5,
                y: 6,
                w: 70,
                h: 80
            },
        ])]
    );
    assert_eq!(
        canonical_key_form("filters:remove_region(5,6,70,80)/example.com/img.jpg"),
        "filters:remove_region(5,6,70,80)/example.com/img.jpg"
    );
    let d = RequestDirectives::analyze(&ops);
    assert!(d.remove_region);
    assert!(
        d.uses_ai(),
        "remove_region runs the inpainting model, so it takes an AI permit"
    );
}

#[test]
fn remove_region_rejects_malformed_rectangles() {
    for seg in [
        "filters:remove_region(1,2,0,40)",
        "filters:remove_region(1,2,30,0)",
        "filters:remove_region(1,2,3)",
        "filters:remove_region(1,2,3,4,5)",
        "filters:remove_region(-1,0,10,10)",
        "filters:remove_region(0,0,10,16384)",
        "filters:remove_region(0,0,10)",
        "filters:remove_region()",
        "filters:remove_region(1.5,2,3,4)",
    ] {
        let path = format!("{seg}/example.com/img.jpg");
        let result = parse_operations(&path);
        assert!(
            matches!(result, Err(ParseError::InvalidFilters(_))),
            "{seg} should fail as invalid filters, got {result:?}"
        );
    }
}

#[test]
fn parses_avif_format() {
    let (ops, _) = parse_operations("filters:format(avif)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
            ImageFormat::Avif
        ))])]
    );
}

#[test]
fn parses_video_formats() {
    for (spelling, format) in [("mp4", ImageFormat::Mp4), ("webm", ImageFormat::WebM)] {
        let (ops, _) =
            parse_operations(&format!("filters:format({spelling})/example.com/img.gif")).unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                format
            ))])]
        );
        assert!(format.is_video());
        // The canonical form spells them back unchanged.
        assert_eq!(
            canonical_form(&ops, "example.com/img.gif"),
            format!("filters:format({spelling})/example.com/img.gif")
        );
    }
    assert!(!ImageFormat::Gif.is_video());
}

#[test]
fn parses_fit_in() {
    let (ops, _) = parse_operations("fit-in:300x200/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::FitIn {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            },
            adaptive: false,
        }]
    );
    let (ops, _) = parse_operations("fit-in:300x0/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::FitIn {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Proportional,
                flip_horizontal: false,
                flip_vertical: false,
            },
            adaptive: false,
        }]
    );
}

#[test]
fn parses_adaptive_fit_in() {
    let (ops, _) = parse_operations("fit-in:300x200:adaptive/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::FitIn {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            },
            adaptive: true,
        }]
    );
    // The modifier is part of the operation: it stays in the
    // canonical form, so an adaptive URL keys apart from the
    // plain fit-in of the same box.
    assert_eq!(
        canonical_form(&ops, "example.com/img.jpg"),
        "fit-in:300x200:adaptive/example.com/img.jpg"
    );
    assert_ne!(
        canonical_key_form("fit-in:300x200:adaptive/example.com/img.jpg"),
        canonical_key_form("fit-in:300x200/example.com/img.jpg")
    );
}

#[test]
fn rejects_bad_fit_in() {
    for seg in [
        "fit-in:-300x200",
        "fit-in:wide",
        "fit-in:",
        // The modifier slot takes exactly `adaptive`, once.
        "fit-in:300x200:sideways",
        "fit-in:300x200:adaptive:extra",
        "fit-in:300x200:",
        "fit-in:300x200:Adaptive",
        // `adaptive` is a modifier, never the box itself.
        "fit-in:adaptive",
        "fit-in:adaptive:300x200",
        // A flipped box stays invalid with the modifier.
        "fit-in:-300x200:adaptive",
    ] {
        let path = format!("{seg}/example.com/img.jpg");
        assert_eq!(
            parse_operations(&path),
            Err(ParseError::InvalidSize(seg.to_string())),
            "{seg}"
        );
    }
}

#[test]
fn parses_crop_rect() {
    let (ops, _) = parse_operations("crop-rect:10x20:110x220/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::CropRect {
            left: 10,
            top: 20,
            right: 110,
            bottom: 220,
        }]
    );
}

#[test]
fn rejects_bad_crop_rect() {
    for seg in [
        "crop-rect:10x20",
        "crop-rect:110x220:10x20",
        "crop-rect:10x20:10x220",
        "crop-rect:10x20:110x20",
        "crop-rect:-1x0:10x10",
        "crop-rect:axbx:cxdx",
    ] {
        let path = format!("{seg}/example.com/img.jpg");
        assert!(
            matches!(parse_operations(&path), Err(ParseError::InvalidCropRect(_))),
            "{seg}"
        );
    }
}

#[test]
fn parses_focal_rect() {
    let (ops, _) = parse_operations("focal:10x20:110x220/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::FocalRect {
            left: 10,
            top: 20,
            right: 110,
            bottom: 220,
        }]
    );
}

#[test]
fn rejects_bad_focal_rect() {
    for seg in [
        "focal:10x20",
        "focal:110x220:10x20",
        "focal:10x20:10x220",
        "focal:10x20:110x20",
        "focal:-1x0:10x10",
        "focal:axbx:cxdx",
    ] {
        let path = format!("{seg}/example.com/img.jpg");
        assert!(
            matches!(
                parse_operations(&path),
                Err(ParseError::InvalidFocalRect(_))
            ),
            "{seg}"
        );
    }
}

#[test]
fn parses_focal_point() {
    // The zoom is optional on input and defaults to 1.
    let (ops, _) = parse_operations("focalpoint:0.75x0.4/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::FocalPoint {
            x: 0.75,
            y: 0.4,
            zoom: 1.0,
        }]
    );
    let (ops, _) = parse_operations("focalpoint:0.75x0.4:2/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::FocalPoint {
            x: 0.75,
            y: 0.4,
            zoom: 2.0,
        }]
    );
    // The corners of the frame are valid points, and the
    // zoom endpoints are in.
    assert!(parse_operations("focalpoint:0x0:100/example.com/img.jpg").is_ok());
    assert!(parse_operations("focalpoint:1x1:0.01/example.com/img.jpg").is_ok());
}

#[test]
fn rejects_bad_focal_point() {
    for seg in [
        "focalpoint:",
        "focalpoint:0.75",
        "focalpoint:0.75x",
        "focalpoint:x0.4",
        "focalpoint:0.75x0.4x0.2",
        "focalpoint:1.5x0.4",
        "focalpoint:0.75x-0.1",
        "focalpoint:nanx0.4",
        "focalpoint:0.75x0.4:0",
        "focalpoint:0.75x0.4:-2",
        "focalpoint:0.75x0.4:101",
        "focalpoint:0.75x0.4:inf",
        "focalpoint:0.75x0.4:2:3",
        "focalpoint:0.75x0.4:",
    ] {
        let path = format!("{seg}/example.com/img.jpg");
        assert!(
            matches!(
                parse_operations(&path),
                Err(ParseError::InvalidFocalPoint(_))
            ),
            "{seg}"
        );
    }
}

#[test]
fn parses_no_upscale() {
    let (ops, _) = parse_operations("filters:no_upscale/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::NoUpscale])]);
}

#[test]
fn parses_watermark() {
    let (ops, _) = parse_operations(
        "filters:watermark(https%3A%2F%2Fcdn.example.com%2Fwm.png,10,-5,80)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Watermark {
            url: "https://cdn.example.com/wm.png".to_string(),
            x: 10,
            y: -5,
            alpha: 80,
        }])]
    );
}

#[test]
fn rejects_bad_watermarks() {
    for seg in [
        "watermark(http%3A%2F%2Fx%2Fwm.png,0,0,101)",
        "watermark(http%3A%2F%2Fx%2Fwm.png,0,0)",
        "watermark(%zz,0,0,50)",
        "watermark(,0,0,50)",
    ] {
        let path = format!("filters:{seg}/example.com/img.jpg");
        assert!(parse_operations(&path).is_err(), "{seg}");
    }
    // `removebg` accepts no parameters.
    assert!(parse_operations("filters:removebg(1)/example.com/img.png").is_err());
}

#[test]
fn watermark_urls_are_distinct_and_ordered() {
    let (ops, _) = parse_operations(
        "filters:watermark(a%2F1.png,0,0,50):watermark(a%2F1.png,5,5,50):watermark(b%2F2.png,0,0,50)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        RequestDirectives::analyze(&ops).watermark_urls,
        ["a/1.png", "b/2.png"]
    );
}

#[test]
fn distinct_watermark_urls_are_capped() {
    // Issue #613: each distinct watermark or overlay URL is one
    // fetch and one decode, so a request may name at most
    // MAX_WATERMARK_URLS of them. They count as the server
    // fetches them, by exact string: a repeat counts once, each
    // `?v=` variant of one object counts again, and watermark
    // and overlay URLs share the cap.
    let parse = |filters: &[String]| {
        parse_operations(&format!(
            "filters:{}/example.com/img.jpg",
            filters.join(":")
        ))
        .unwrap()
        .0
    };
    let variant = |i: usize| {
        if i.is_multiple_of(2) {
            format!("watermark(wm.png%3Fv%3D{i},0,0,50)")
        } else {
            format!("overlay(wm.png%3Fv%3D{i},0,0,50,normal)")
        }
    };
    let at_cap: Vec<String> = (0..MAX_WATERMARK_URLS).map(variant).collect();
    assert_eq!(
        RequestDirectives::analyze(&parse(&at_cap))
            .watermark_urls
            .len(),
        MAX_WATERMARK_URLS
    );
    assert!(check_watermark_count(&parse(&at_cap)).is_ok());
    let twice: Vec<String> = at_cap.iter().chain(&at_cap).cloned().collect();
    assert!(check_watermark_count(&parse(&twice)).is_ok());
    let mut over = at_cap;
    over.push(variant(MAX_WATERMARK_URLS));
    let err = check_watermark_count(&parse(&over)).unwrap_err();
    assert_eq!(err, ParseError::TooManyWatermarks(MAX_WATERMARK_URLS + 1));
    assert_eq!(err.code(), "unsupported_operation");
    assert_eq!(
        err.to_string(),
        "request names 17 distinct watermark and overlay URLs, over the 16 URL limit"
    );
    // A request with no watermark passes.
    assert!(check_watermark_count(&parse(&["invert".to_string()])).is_ok());
}

#[test]
fn parses_trim() {
    let (ops, _) = parse_operations("filters:trim/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Trim(10)])]);
    let (ops, _) = parse_operations("filters:trim(0)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Trim(0)])]);
    assert!(parse_operations("filters:trim(256)/example.com/img.jpg").is_err());
    assert!(parse_operations("filters:trim(-1)/example.com/img.jpg").is_err());
}

#[test]
fn parses_trim_with_per_side_amounts() {
    let (ops, _) = parse_operations("filters:trim(20,5,0,5,0)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::TrimSides(
            20,
            [5, 0, 5, 0]
        )])]
    );
    // The largest side the grammar takes.
    assert!(parse_operations("filters:trim(0,4294967295,0,0,0)/example.com/img.jpg").is_ok());
    // Partial forms are malformed, not shorthand.
    for seg in [
        "filters:trim(20,5)",
        "filters:trim(20,5,0)",
        "filters:trim(20,5,0,5)",
        "filters:trim(20,5,0,5,0,1)",
        "filters:trim(20,)",
        "filters:trim(,5,0,5,0)",
    ] {
        assert!(
            parse_operations(&format!("{seg}/example.com/img.jpg")).is_err(),
            "{seg}"
        );
    }
    // The tolerance keeps its own bound inside the
    // five-argument form, and the sides are unsigned.
    for seg in [
        "filters:trim(256,0,0,0,0)",
        "filters:trim(20,-1,0,0,0)",
        "filters:trim(20,4294967296,0,0,0)",
        "filters:trim(20,1.5,0,0,0)",
    ] {
        assert!(
            parse_operations(&format!("{seg}/example.com/img.jpg")).is_err(),
            "{seg}"
        );
    }
}

#[test]
fn parses_gif_format() {
    let (ops, _) = parse_operations("filters:format(gif)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
            ImageFormat::Gif
        ))])]
    );
}

#[test]
fn parses_crop_with_smart_alignment() {
    let (ops, _) = parse_operations("crop:300x200:smart/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: Align::Smart,
        }]
    );
}

#[test]
fn parses_face_operations() {
    let (ops, _) = parse_operations("crop:300x300:face/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(300),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: Align::Face,
        }]
    );
    assert!(RequestDirectives::analyze(&ops).needs_face_model);
    assert_eq!(RequestDirectives::analyze(&ops).face_padding, 0.5);

    let (ops, _) =
        parse_operations("filters:blurfaces:pixelatefaces:facepad(1.5)/example.com/img.jpg")
            .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::BlurFaces,
            Filter::PixelateFaces,
            Filter::FacePad(1.5),
        ])]
    );
    assert!(RequestDirectives::analyze(&ops).needs_face_model);
    assert_eq!(RequestDirectives::analyze(&ops).face_padding, 1.5);

    // Smart crops and plain requests do not require a model.
    let (ops, _) = parse_operations("crop:300x300:smart/example.com/img.jpg").unwrap();
    assert!(!RequestDirectives::analyze(&ops).needs_face_model);

    // Class redaction (#62): names parse as written,
    // spaces included; the class list is in order.
    let (ops, _) = parse_operations(
        "filters:blurclass(person):pixelateclass(traffic%20light)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::BlurClass("person".to_string()),
            Filter::PixelateClass("traffic light".to_string()),
        ])]
    );
    assert!(RequestDirectives::analyze(&ops).has_class_redaction());
    assert_eq!(
        RequestDirectives::analyze(&ops).classes,
        ["person", "traffic light"]
    );
    for bad in [
        "filters:blurclass()/example.com/img.jpg",
        "filters:blurclass( )/example.com/img.jpg",
        "filters:blurclass(per%24on)/example.com/img.jpg",
        "filters:pixelateclass/example.com/img.jpg",
    ] {
        assert!(parse_operations(bad).is_err(), "{bad}");
    }
    for bad in [
        "filters:facepad(9)/example.com/img.jpg",
        "filters:facepad(-1)/example.com/img.jpg",
        "filters:facepad(x)/example.com/img.jpg",
        "filters:blurfaces(1)/example.com/img.jpg",
    ] {
        assert!(parse_operations(bad).is_err(), "{bad}");
    }
}

#[test]
fn parses_objects_crop_alignment() {
    // Issue #318: one class, the scale defaulted.
    let (ops, _) = parse_operations("crop:300x200:objects(person)/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: Align::Objects {
                classes: vec!["person".to_string()],
                scale: 1.0,
            },
        }]
    );
    // The mode takes the detection preflight and the AI
    // gate, through its own directive fields: it is not
    // class redaction and not a face operation.
    let d = RequestDirectives::analyze(&ops);
    assert!(d.objects_crop);
    assert_eq!(d.object_classes, ["person"]);
    assert!(d.uses_ai());
    assert!(!d.has_class_redaction());
    assert!(!d.needs_face_model);

    // Several classes and an explicit scale; class names
    // percent-decode like the class filters' argument.
    let (ops, _) = parse_operations(
        "crop:300x200:objects(person,traffic%20light,car):1.5/example.com/img.jpg",
    )
    .unwrap();
    assert!(matches!(
        &ops[0],
        Operation::Crop {
            align: Align::Objects { classes, scale },
            ..
        } if classes == &["person", "traffic light", "car"] && *scale == 1.5
    ));
    let d = RequestDirectives::analyze(&ops);
    assert_eq!(d.object_classes, ["person", "traffic light", "car"]);

    // A class named twice names one class: the first
    // spelling stands (matching itself is
    // case-insensitive, like the vocabulary checks).
    let (ops, _) =
        parse_operations("crop:300x200:objects(Person,person,car)/example.com/img.jpg").unwrap();
    assert!(matches!(
        &ops[0],
        Operation::Crop {
            align: Align::Objects { classes, .. },
            ..
        } if classes == &["Person", "car"]
    ));
}

#[test]
fn objects_crop_canonical_form() {
    // The scale is optional on input and always spelled
    // in the canonical form, the focalpoint zoom rule;
    // class names re-encode deterministically.
    assert_eq!(
        canonical_key_form("crop:300x200:objects(person)/example.com/img.jpg"),
        "crop:300x200:objects(person):1/example.com/img.jpg"
    );
    assert_eq!(
        canonical_key_form("crop:300x200:objects(person):1.50/example.com/img.jpg"),
        "crop:300x200:objects(person):1.5/example.com/img.jpg"
    );
    assert_eq!(
        canonical_key_form("crop:300x200:objects(traffic%20light,person):2/example.com/img.jpg"),
        "crop:300x200:objects(traffic%20light,person):2/example.com/img.jpg"
    );
    // The canonical form parses back to the same ops.
    let (ops, _) =
        parse_operations("crop:300x200:objects(traffic%20light,person):2/example.com/img.jpg")
            .unwrap();
    assert!(matches!(
        &ops[0],
        Operation::Crop {
            align: Align::Objects { classes, scale },
            ..
        } if classes == &["traffic light", "person"] && *scale == 2.0
    ));
}

#[test]
fn rejects_bad_objects_crops() {
    for bad in [
        // No classes, or an empty one.
        "crop:300x200:objects()/example.com/img.jpg",
        "crop:300x200:objects( )/example.com/img.jpg",
        "crop:300x200:objects(person,)/example.com/img.jpg",
        "crop:300x200:objects(,person)/example.com/img.jpg",
        // The class charset is the class filters' own.
        "crop:300x200:objects(per%24on)/example.com/img.jpg",
        "crop:300x200:objects(per%2Con)/example.com/img.jpg",
        // The parameterized form must be complete.
        "crop:300x200:objects/example.com/img.jpg",
        "crop:300x200:objects(person/example.com/img.jpg",
        "crop:300x200:objects(person)junk/example.com/img.jpg",
        "crop:300x200:objects(person):/example.com/img.jpg",
        "crop:300x200:objects(person):1.5:extra/example.com/img.jpg",
        // The scale is a positive multiplier up to the cap.
        "crop:300x200:objects(person):0/example.com/img.jpg",
        "crop:300x200:objects(person):-1/example.com/img.jpg",
        "crop:300x200:objects(person):100.5/example.com/img.jpg",
        "crop:300x200:objects(person):NaN/example.com/img.jpg",
        "crop:300x200:objects(person):inf/example.com/img.jpg",
        "crop:300x200:objects(person):wide/example.com/img.jpg",
        // The exact-box contract still applies.
        "crop:300:objects(person)/example.com/img.jpg",
        "crop:0x200:objects(person)/example.com/img.jpg",
        "crop:-300x200:objects(person)/example.com/img.jpg",
    ] {
        assert!(parse_operations(bad).is_err(), "{bad}");
    }
    // The class list is bounded: 32 names parse in
    // full.
    let names = |n: usize| {
        (0..n)
            .map(|i| format!("c{i}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let ok = format!("crop:300x200:objects({})/example.com/img.jpg", names(32));
    let (ops, _) = parse_operations(&ok).unwrap();
    assert!(matches!(
        &ops[0],
        Operation::Crop {
            align: Align::Objects { classes, .. },
            ..
        } if classes.len() == MAX_OBJECTS_CLASSES
            && classes.first().map(String::as_str) == Some("c0")
            && classes.last().map(String::as_str) == Some("c31")
    ));
    // A 33rd distinct class is a bad crop (issue #408),
    // as it was at 0.9.10, not a silent truncation to
    // the first 32 (what the #391 fix briefly shipped).
    for n in [33, 40] {
        let over = format!("crop:300x200:objects({})/example.com/img.jpg", names(n));
        assert!(
            matches!(parse_operations(&over), Err(ParseError::InvalidCrop(_))),
            "{n} classes"
        );
    }
    // The cap counts distinct names: repeats in any case
    // fold before they count, so 32 classes spelled 64
    // times parse to the 32.
    let repeated = format!(
        "crop:300x200:objects({},{})/example.com/img.jpg",
        names(32),
        names(32).to_uppercase()
    );
    let (ops, _) = parse_operations(&repeated).unwrap();
    assert!(matches!(
        &ops[0],
        Operation::Crop {
            align: Align::Objects { classes, .. },
            ..
        } if classes.len() == MAX_OBJECTS_CLASSES
    ));
    // An invalid name past a full set is still a bad crop.
    let bad_tail = format!(
        "crop:300x200:objects({},per%24on)/example.com/img.jpg",
        names(32)
    );
    assert!(parse_operations(&bad_tail).is_err());
}

#[test]
fn a_long_objects_list_is_rejected_at_the_cap() {
    // Hostile input (issues #391, #408): a list of
    // thousands of distinct names is rejected when the
    // 33rd arrives, so the parse never scans past a
    // cap-sized set; one of thousands of repeats of a
    // single name parses to that one class. Both stay
    // linear in the segment, which the URI length
    // already bounds.
    let distinct: Vec<String> = (0..20_000).map(|i| format!("c{i}")).collect();
    let over = format!(
        "crop:300x200:objects({})/example.com/img.jpg",
        distinct.join(",")
    );
    assert!(matches!(
        parse_operations(&over),
        Err(ParseError::InvalidCrop(_))
    ));
    let same = vec!["person"; 20_000].join(",");
    let (ops, _) =
        parse_operations(&format!("crop:300x200:objects({same})/example.com/img.jpg")).unwrap();
    assert!(matches!(
        &ops[0],
        Operation::Crop {
            align: Align::Objects { classes, .. },
            ..
        } if classes.len() == 1
    ));
}

#[test]
fn recolorclass_parses_like_the_other_class_filters() {
    // Issue #175: <class>,<hex>, the class percent-
    // decoded like the other class filters, the hex in
    // the color parameters' spelling; the canonical
    // form lowercases the hex and re-encodes the name.
    let (ops, _) =
        parse_operations("filters:recolorclass(traffic%20light,FF8800)/example.com/img.jpg")
            .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![Filter::RecolorClass(
            "traffic light".to_string(),
            [0xff, 0x88, 0x00],
        )])]
    );
    assert_eq!(
        canonical_key_form("filters:recolorclass(traffic%20light,FF8800)/example.com/img.jpg"),
        "filters:recolorclass(traffic%20light,ff8800)/example.com/img.jpg"
    );
    // A recolor names a class like the redaction
    // filters: it joins the class list and takes the
    // AI gate.
    let d = RequestDirectives::analyze(&ops);
    assert_eq!(d.classes, ["traffic light"]);
    assert!(d.has_class_redaction());
    assert!(d.uses_ai());
    for bad in [
        "filters:recolorclass(person)/example.com/img.jpg",
        "filters:recolorclass(person,)/example.com/img.jpg",
        "filters:recolorclass(,ff0000)/example.com/img.jpg",
        "filters:recolorclass(person,ff00)/example.com/img.jpg",
        "filters:recolorclass(person,ff00000)/example.com/img.jpg",
        "filters:recolorclass(person,gg0000)/example.com/img.jpg",
        "filters:recolorclass(person,#ff0000)/example.com/img.jpg",
        "filters:recolorclass(per%24on,ff0000)/example.com/img.jpg",
        "filters:recolorclass/example.com/img.jpg",
    ] {
        assert!(parse_operations(bad).is_err(), "{bad}");
    }
}

#[test]
fn parses_pad() {
    let size = ImageSize {
        width: Dimension::Pixels(200),
        height: Dimension::Pixels(100),
        flip_horizontal: false,
        flip_vertical: false,
    };
    let (ops, _) = parse_operations("pad:200x100/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Pad {
            size,
            fill: PadFill::Directive
        }]
    );
    let (ops, _) = parse_operations("pad:200x100:blur/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Pad {
            size,
            fill: PadFill::Blur
        }]
    );
    let (ops, _) = parse_operations("pad:200x100:ff8800/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![Operation::Pad {
            size,
            fill: PadFill::Color([255, 136, 0])
        }]
    );
    for bad in [
        "pad:",
        "pad:200",
        "pad:200x100:red",
        "pad:200x100:ff88",
        "pad:200x100:blur:extra",
        // `adaptive` belongs to fit-in: pad's canvas is
        // explicit, so the modifier is not a pad fill.
        "pad:200x100:adaptive",
        // The exact-box contract, same as crop:.
        "pad:200x0",
        "pad:origx100",
        "pad:-200x100",
    ] {
        assert!(parse_operations(bad).is_err(), "{bad}");
    }
}

#[test]
fn parses_expiry_segments() {
    let (ops, src, exp, _) =
        parse_operations_with_expiry("exp:1893456000/300x200/example.com/img.jpg").unwrap();
    assert_eq!(exp, Some(1893456000));
    assert_eq!(ops.len(), 1);
    assert_eq!(src, "example.com/img.jpg");
    // Position-independent among the operations; last wins.
    let (_, _, exp, _) =
        parse_operations_with_expiry("300x200/exp:100/exp:200/example.com/img.jpg").unwrap();
    assert_eq!(exp, Some(200));
    // Once the source URL starts, `exp:` is just URL text.
    let (ops, src, exp, _) = parse_operations_with_expiry("example.com/exp:123").unwrap();
    assert!(ops.is_empty() && exp.is_none());
    assert_eq!(src, "example.com/exp:123");
    // Absent means no expiry.
    let (_, _, exp, _) = parse_operations_with_expiry("300x200/example.com/img.jpg").unwrap();
    assert_eq!(exp, None);
    // Malformed values are parse errors.
    for bad in [
        "exp:/example.com/i.jpg",
        "exp:abc/example.com/i.jpg",
        "exp:-1/example.com/i.jpg",
        "exp:12.5/example.com/i.jpg",
        "exp:99999999999999999999999999/example.com/i.jpg",
    ] {
        assert!(parse_operations_with_expiry(bad).is_err(), "{bad}");
    }
}

#[test]
fn parses_maxage_segments() {
    let (ops, src, _, max_age) =
        parse_operations_with_expiry("maxage:60/300x200/example.com/img.jpg").unwrap();
    assert_eq!(max_age, Some(60));
    assert_eq!(ops.len(), 1);
    assert_eq!(src, "example.com/img.jpg");
    // Position-independent among the operations; last wins.
    let (_, _, _, max_age) =
        parse_operations_with_expiry("300x200/maxage:100/maxage:200/example.com/img.jpg").unwrap();
    assert_eq!(max_age, Some(200));
    // Zero is valid (revalidate every time), and so is the
    // one-year cap itself.
    let (_, _, _, max_age) = parse_operations_with_expiry("maxage:0/example.com/img.jpg").unwrap();
    assert_eq!(max_age, Some(0));
    let (_, _, _, max_age) =
        parse_operations_with_expiry("maxage:31536000/example.com/img.jpg").unwrap();
    assert_eq!(max_age, Some(MAX_CACHE_AGE));
    // Once the source URL starts, `maxage:` is just URL text.
    let (ops, src, _, max_age) = parse_operations_with_expiry("example.com/maxage:60").unwrap();
    assert!(ops.is_empty() && max_age.is_none());
    assert_eq!(src, "example.com/maxage:60");
    // Absent means the server default.
    let (_, _, _, max_age) = parse_operations_with_expiry("300x200/example.com/img.jpg").unwrap();
    assert_eq!(max_age, None);
    // Malformed and out-of-range values are parse errors of
    // the dedicated kind.
    for bad in [
        "maxage:/example.com/i.jpg",
        "maxage:abc/example.com/i.jpg",
        "maxage:-1/example.com/i.jpg",
        "maxage:12.5/example.com/i.jpg",
        "maxage:31536001/example.com/i.jpg",
        "maxage:99999999999999999999999999/example.com/i.jpg",
    ] {
        assert!(
            matches!(
                parse_operations_with_expiry(bad),
                Err(ParseError::InvalidMaxAge(_))
            ),
            "{bad}"
        );
    }
    // The two metadata kinds extract side by side.
    let (_, _, exp, max_age) =
        parse_operations_with_expiry("exp:1893456000/maxage:60/300x200/example.com/img.jpg")
            .unwrap();
    assert_eq!(exp, Some(1893456000));
    assert_eq!(max_age, Some(60));
}

#[test]
fn parses_debug_focal_segments() {
    let (ops, src, _, _) =
        parse_operations_with_expiry("debug:focal/crop:100x100:smart/example.com/img.jpg").unwrap();
    assert_eq!(
        ops,
        vec![
            Operation::DebugFocal,
            Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(100),
                    height: Dimension::Pixels(100),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Smart,
            },
        ]
    );
    assert_eq!(src, "example.com/img.jpg");
    // The directive analysis sees it.
    assert!(RequestDirectives::analyze(&ops).debug_focal);
    // It is an operation, not metadata: it stays in the
    // canonical form in place, so a debug URL keys apart
    // from the same URL without it (and the server bypasses
    // the cache for it regardless).
    let with = canonical_key_form("debug:focal/crop:100x100:smart/example.com/img.jpg");
    let without = canonical_key_form("crop:100x100:smart/example.com/img.jpg");
    assert_eq!(with, "debug:focal/crop:100x100:smart/example.com/img.jpg");
    assert_ne!(with, without);
    // Once the source URL starts, `debug:` is just URL text.
    let (ops, src, _, _) = parse_operations_with_expiry("example.com/debug:focal").unwrap();
    assert!(ops.is_empty());
    assert_eq!(src, "example.com/debug:focal");
    // Any other debug view is a parse error of the
    // dedicated kind.
    for bad in [
        "debug:/example.com/i.jpg",
        "debug:banana/example.com/i.jpg",
        "debug:focalx/example.com/i.jpg",
        "debug:smart/example.com/i.jpg",
    ] {
        let err = parse_operations_with_expiry(bad).unwrap_err();
        assert!(matches!(err, ParseError::InvalidDebug(_)), "{bad}");
        assert_eq!(err.code(), "invalid_debug");
    }
}

#[test]
fn frame_segments_parse_and_canonicalize() {
    // The three selector forms (issue #214).
    let (ops, src, _, _) = parse_operations_with_expiry("frame:poster/example.com/v.mp4").unwrap();
    assert_eq!(ops, vec![Operation::Frame(FrameSelection::Poster)]);
    assert_eq!(src, "example.com/v.mp4");
    let (ops, ..) = parse_operations_with_expiry("frame:12.5/example.com/v.mp4").unwrap();
    assert_eq!(ops, vec![Operation::Frame(FrameSelection::Seconds(12.5))]);
    let (ops, ..) = parse_operations_with_expiry("frame:50%/example.com/v.mp4").unwrap();
    assert_eq!(ops, vec![Operation::Frame(FrameSelection::Percent(50.0))]);
    // Bounds are inclusive: zero, the full duration, and
    // the 24 h cap itself all parse.
    for (seg, want) in [
        ("frame:0", FrameSelection::Seconds(0.0)),
        ("frame:100%", FrameSelection::Percent(100.0)),
        ("frame:0%", FrameSelection::Percent(0.0)),
        ("frame:86400", FrameSelection::Seconds(86_400.0)),
        ("frame:33.333%", FrameSelection::Percent(33.333)),
    ] {
        let (ops, ..) = parse_operations_with_expiry(&format!("{seg}/example.com/v.mp4"))
            .unwrap_or_else(|e| panic!("{seg}: {e}"));
        assert_eq!(ops, vec![Operation::Frame(want)], "{seg}");
    }
    // Canonical spelling: whole values print as integers,
    // fractions shorten, so spelling variants are one
    // request with one cache entry.
    assert_eq!(
        canonical_key_form("frame:1.50/300x200/example.com/v.mp4"),
        "frame:1.5/300x200/example.com/v.mp4"
    );
    assert_eq!(
        canonical_key_form("frame:2.0/example.com/v.mp4"),
        "frame:2/example.com/v.mp4"
    );
    assert_eq!(
        canonical_key_form("frame:050%/example.com/v.mp4"),
        "frame:50%/example.com/v.mp4"
    );
    // The directive analysis sees the selection, and the
    // last segment wins when a request carries several.
    let (ops, ..) = parse_operations_with_expiry("frame:1/frame:poster/example.com/v.mp4").unwrap();
    assert_eq!(
        RequestDirectives::analyze(&ops).frame,
        Some(FrameSelection::Poster)
    );
    let (ops, ..) = parse_operations_with_expiry("300x200/example.com/v.mp4").unwrap();
    assert_eq!(RequestDirectives::analyze(&ops).frame, None);
    // A frame segment inside the source is URL text.
    let (ops, src, _, _) = parse_operations_with_expiry("example.com/frame:1/v.mp4").unwrap();
    assert!(ops.is_empty());
    assert_eq!(src, "example.com/frame:1/v.mp4");
}

#[test]
fn frame_segments_refuse_malformed_selectors() {
    for bad in [
        "frame:/example.com/v.mp4",
        "frame:banana/example.com/v.mp4",
        "frame:-1/example.com/v.mp4",
        "frame:+1/example.com/v.mp4",
        "frame:.5/example.com/v.mp4",
        "frame:1./example.com/v.mp4",
        "frame:1.2.3/example.com/v.mp4",
        "frame:1e3/example.com/v.mp4",
        "frame:86401/example.com/v.mp4",
        "frame:86400.5/example.com/v.mp4",
        "frame:101%/example.com/v.mp4",
        "frame:100.1%/example.com/v.mp4",
        "frame:%/example.com/v.mp4",
        "frame:50%%/example.com/v.mp4",
        "frame:poster%/example.com/v.mp4",
        "frame:Poster/example.com/v.mp4",
    ] {
        let err = parse_operations_with_expiry(bad).unwrap_err();
        assert!(matches!(err, ParseError::InvalidFrame(_)), "{bad}: {err:?}");
        assert_eq!(err.code(), "invalid_frame", "{bad}");
    }
}

#[test]
fn page_segments_parse_and_canonicalize() {
    // The selector form (issue #215): 1-based digits.
    let (ops, src, _, _) = parse_operations_with_expiry("page:1/example.com/doc.pdf").unwrap();
    assert_eq!(ops, vec![Operation::Page(1)]);
    assert_eq!(src, "example.com/doc.pdf");
    // Bounds are inclusive: the first page and the
    // grammar bound itself both parse.
    for (seg, want) in [("page:1", 1), ("page:999999", 999_999), ("page:42", 42)] {
        let (ops, ..) = parse_operations_with_expiry(&format!("{seg}/example.com/doc.pdf"))
            .unwrap_or_else(|e| panic!("{seg}: {e}"));
        assert_eq!(ops, vec![Operation::Page(want)], "{seg}");
    }
    // Canonical spelling: leading zeros are spelling,
    // not meaning, so page:007 and page:7 are one
    // request with one cache entry.
    assert_eq!(
        canonical_key_form("page:007/300x200/example.com/doc.pdf"),
        "page:7/300x200/example.com/doc.pdf"
    );
    assert_eq!(
        canonical_key_form("page:000001/example.com/doc.pdf"),
        "page:1/example.com/doc.pdf"
    );
    // The directive analysis sees the selection, and the
    // last segment wins when a request carries several.
    let (ops, ..) = parse_operations_with_expiry("page:2/page:5/example.com/doc.pdf").unwrap();
    assert_eq!(RequestDirectives::analyze(&ops).page, Some(5));
    let (ops, ..) = parse_operations_with_expiry("300x200/example.com/doc.pdf").unwrap();
    assert_eq!(RequestDirectives::analyze(&ops).page, None);
    // A page segment inside the source is URL text.
    let (ops, src, _, _) = parse_operations_with_expiry("example.com/page:2/doc.pdf").unwrap();
    assert!(ops.is_empty());
    assert_eq!(src, "example.com/page:2/doc.pdf");
}

#[test]
fn page_segments_refuse_malformed_selectors() {
    for bad in [
        "page:/example.com/doc.pdf",
        "page:0/example.com/doc.pdf",
        "page:00/example.com/doc.pdf",
        "page:-1/example.com/doc.pdf",
        "page:+1/example.com/doc.pdf",
        "page:1.5/example.com/doc.pdf",
        "page:1./example.com/doc.pdf",
        "page:.5/example.com/doc.pdf",
        "page:1e3/example.com/doc.pdf",
        "page:abc/example.com/doc.pdf",
        "page:1000000/example.com/doc.pdf",
        "page:9999999/example.com/doc.pdf",
        "page: 1/example.com/doc.pdf",
        "page:1 /example.com/doc.pdf",
    ] {
        let err = parse_operations_with_expiry(bad).unwrap_err();
        assert!(matches!(err, ParseError::InvalidPage(_)), "{bad}: {err:?}");
        assert_eq!(err.code(), "invalid_page", "{bad}");
    }
}

#[test]
fn debug_focal_requests_count_as_ai_use() {
    // `debug:focal` runs the face and focal providers
    // and is never cached, so the directive analysis
    // classes it as AI use: the server holds an AI-gate
    // permit for it (audit R2-17).
    let (ops, _, _, _) = parse_operations_with_expiry("debug:focal/example.com/i.jpg").unwrap();
    assert!(RequestDirectives::analyze(&ops).uses_ai());
    // A plain transform is not AI use.
    let (ops, _, _, _) = parse_operations_with_expiry("100x100/example.com/i.jpg").unwrap();
    assert!(!RequestDirectives::analyze(&ops).uses_ai());
}

fn preset_map() -> Presets {
    [
        ("thumb".to_string(), "300x200".parse().unwrap()),
        (
            "og".to_string(),
            "crop:1200x630:smart/filters:format(webp)".parse().unwrap(),
        ),
    ]
    .into_iter()
    .collect()
}

#[test]
fn preset_definitions_validate() {
    assert_eq!(
        parse_preset_operations("crop:1200x630:smart/filters:format(webp):quality(80)")
            .unwrap()
            .len(),
        2
    );
    // Source-like segments, expiry, cache-lifetime metadata,
    // nested presets, and the empty string are all invalid
    // definitions.
    for bad in [
        "",
        "example.com/img.jpg",
        "exp:123/300x200",
        "maxage:60/300x200",
        "preset:other/300x200",
        "300x200/",
    ] {
        assert!(parse_preset_operations(bad).is_err(), "{bad}");
    }
}

#[test]
fn presets_expand_in_the_operations_region() {
    let map = preset_map();
    assert_eq!(
        expand_presets("preset:thumb/example.com/img.jpg", &map, false).unwrap(),
        "300x200/example.com/img.jpg"
    );
    assert_eq!(
        expand_presets("preset:og/exp:999/example.com/img.jpg", &map, false).unwrap(),
        "crop:1200x630:smart/filters:format(webp)/exp:999/example.com/img.jpg"
    );
    // maxage: metadata walks like exp: it neither stops the
    // expansion nor is expanded itself, on either side of a
    // preset reference.
    assert_eq!(
        expand_presets("maxage:60/preset:thumb/example.com/img.jpg", &map, false).unwrap(),
        "maxage:60/300x200/example.com/img.jpg"
    );
    assert_eq!(
        expand_presets("preset:thumb/maxage:60/example.com/img.jpg", &map, false).unwrap(),
        "300x200/maxage:60/example.com/img.jpg"
    );
    // Mixed with free-form operations.
    assert_eq!(
        expand_presets(
            "preset:thumb/filters:grayscale/example.com/i.jpg",
            &map,
            false
        )
        .unwrap(),
        "300x200/filters:grayscale/example.com/i.jpg"
    );
    // A preset: segment inside the source path is URL text.
    assert_eq!(
        expand_presets("example.com/preset:thumb/img.jpg", &map, false).unwrap(),
        "example.com/preset:thumb/img.jpg"
    );
    // Unknown names are errors.
    assert_eq!(
        expand_presets("preset:nope/example.com/img.jpg", &map, false),
        Err(ParseError::UnknownPreset("nope".to_string()))
    );
}

fn root_map() -> std::collections::HashMap<String, String> {
    [
        (
            "photos".to_string(),
            "https://cdn.example.com/photos".to_string(),
        ),
        (
            "media".to_string(),
            "https://assets.example.com".to_string(),
        ),
    ]
    .into_iter()
    .collect()
}

#[test]
fn root_sources_expand_against_the_named_root() {
    let map = root_map();
    let (expanded, reference) = expand_source_root("root:photos/cat.jpg", &map).unwrap();
    assert_eq!(expanded, "https://cdn.example.com/photos/cat.jpg");
    assert_eq!(
        reference,
        Some(RootRef {
            name: "photos".to_string(),
            path: "cat.jpg".to_string(),
        })
    );
    // A root with no path prefix joins at the host root.
    let (expanded, _) = expand_source_root("root:media/a/b.png", &map).unwrap();
    assert_eq!(expanded, "https://assets.example.com/a/b.png");
    // Non-root sources pass through untouched.
    for source in [
        "https://example.com/cat.jpg",
        "s3://media/cat.jpg",
        "file:///srv/cat.jpg",
        "example.com/cat.jpg",
        "https://example.com/root:photos/cat.jpg",
    ] {
        assert_eq!(
            expand_source_root(source, &map).unwrap(),
            (source.to_string(), None),
            "{source}"
        );
    }
}

#[test]
fn root_sources_answer_unknown_root_for_any_unconfigured_name() {
    let map = root_map();
    assert_eq!(
        expand_source_root("root:nope/cat.jpg", &map),
        Err(ParseError::UnknownRoot("nope".to_string()))
    );
    // With no roots configured every name is unknown, the
    // same rule presets follow.
    let empty = std::collections::HashMap::new();
    assert_eq!(
        expand_source_root("root:photos/cat.jpg", &empty),
        Err(ParseError::UnknownRoot("photos".to_string()))
    );
    assert_eq!(
        ParseError::UnknownRoot("x".to_string()).code(),
        "unknown_root"
    );
    assert_eq!(
        ParseError::InvalidRoot("x".to_string()).code(),
        "invalid_root"
    );
}

#[test]
fn root_paths_normalize_dots_but_never_escape_the_root() {
    let map = root_map();
    for (path, joined) in [
        ("a/./cat.jpg", "a/cat.jpg"),
        ("a/b/../cat.jpg", "a/cat.jpg"),
        ("./cat.jpg", "cat.jpg"),
        ("a//cat.jpg", "a//cat.jpg"),
    ] {
        let (expanded, _) = expand_source_root(&format!("root:photos/{path}"), &map).unwrap();
        assert_eq!(expanded, format!("https://cdn.example.com/photos/{joined}"));
    }
    for bad in [
        "root:photos",
        "root:photos/",
        "root:/cat.jpg",
        "root:photos/../secret.jpg",
        "root:photos/a/../../secret.jpg",
        "root:photos/%2e%2e/secret.jpg",
        "root:photos/%2E./secret.jpg",
        "root:photos/a/%2e%2e/%2e%2e/secret.jpg",
        "root:photos/..%2fsecret.jpg",
        "root:photos/a%2fb.jpg",
        "root:photos/..\\secret.jpg",
        "root:photos/a/..",
    ] {
        assert_eq!(
            expand_source_root(bad, &map),
            Err(ParseError::InvalidRoot(bad.to_string())),
            "{bad}"
        );
    }
}

#[test]
fn root_expansion_inside_a_remainder_matches_the_serve_parse() {
    let map = root_map();
    assert_eq!(
        expand_source_root_in_remainder("300x200/root:photos/cat.jpg", &map).unwrap(),
        "300x200/https://cdn.example.com/photos/cat.jpg"
    );
    // exp: metadata in front of the source does not
    // disturb the rewrite, and the source's own text
    // after the root segment stays intact.
    assert_eq!(
        expand_source_root_in_remainder("exp:999/root:media/a/root:b.jpg", &map).unwrap(),
        "exp:999/https://assets.example.com/a/root:b.jpg"
    );
    // Full sources and unparseable remainders pass
    // through unchanged.
    assert_eq!(
        expand_source_root_in_remainder("300x200/https://example.com/cat.jpg", &map).unwrap(),
        "300x200/https://example.com/cat.jpg"
    );
    assert_eq!(
        expand_source_root_in_remainder("root:nope/cat.jpg", &map),
        Err(ParseError::UnknownRoot("nope".to_string()))
    );
}

/// Issue #250: any name that is not exactly a configured
/// key is `unknown_root`, however it is spelled. Names
/// are never decoded, normalized, or case-folded before
/// the lookup, so lookalikes cannot reach a configured
/// root.
#[test]
fn root_names_that_are_not_the_configured_key_are_unknown_root() {
    let map = root_map();
    for (source, name) in [
        ("root:Photos/cat.jpg", "Photos"),
        ("root:PHOTOS/cat.jpg", "PHOTOS"),
        ("root:photo/cat.jpg", "photo"),
        ("root:photoss/cat.jpg", "photoss"),
        ("root:photos./cat.jpg", "photos."),
        ("root:photos name/cat.jpg", "photos name"),
        ("root: photos/cat.jpg", " photos"),
        ("root:photos%2f/cat.jpg", "photos%2f"),
        ("root:photos:80/cat.jpg", "photos:80"),
        ("root:root:photos/cat.jpg", "root:photos"),
        ("root:fötös/cat.jpg", "fötös"),
        ("root:🦀/cat.jpg", "🦀"),
    ] {
        let err = expand_source_root(source, &map).unwrap_err();
        assert_eq!(err, ParseError::UnknownRoot(name.to_string()), "{source}");
        assert_eq!(err.code(), "unknown_root", "{source}");
    }
    // A name of arbitrary length is a lookup miss, not a
    // failure mode of its own.
    let long_name = "a".repeat(10_000);
    let source = format!("root:{long_name}/cat.jpg");
    assert_eq!(
        expand_source_root(&source, &map),
        Err(ParseError::UnknownRoot(long_name))
    );
}

/// Issue #250: short forms that cannot name a root plus
/// a path under it are `invalid_root`: the shape check
/// runs before the name lookup, so even a configured
/// name cannot rescue a malformed spelling.
#[test]
fn root_short_forms_without_a_usable_path_are_invalid_root() {
    let map = root_map();
    for source in [
        "root:",                   // nothing after the prefix
        "root:/",                  // a slash and nothing else
        "root://cat.jpg",          // empty name
        "root:photos//",           // path is one empty segment
        "root:photos///",          // path is only empty segments
        "root:photos/.",           // path normalizes to nothing
        "root:photos/%2e",         // an encoded dot, the same
        "root:photos/%2e/",        // dot and an empty segment
        "root:photos/%",           // malformed percent escape
        "root:photos/%zz/cat.jpg", // malformed percent escape
        "root:photos/cat.jpg%",    // truncated escape at the end
        "root:photos/%ff/cat.jpg", // escape to invalid UTF-8
    ] {
        let err = expand_source_root(source, &map).unwrap_err();
        assert_eq!(err, ParseError::InvalidRoot(source.to_string()), "{source}");
        assert_eq!(err.code(), "invalid_root", "{source}");
    }
}

/// Issue #250: traversal and host-change tricks. What
/// can be refused is refused; what cannot be refused
/// (it is a legal path under the root) still expands
/// to a URL under the root's base, byte for byte.
#[test]
fn root_paths_cannot_escape_by_encoding_or_host_tricks() {
    let map = root_map();
    for bad in [
        "root:photos/..%5csecret.jpg",      // encoded backslash climb
        "root:photos/%2e%2e%5csecret.jpg",  // encoded dots, encoded backslash
        "root:photos/a/%5c..%5csecret.jpg", // separators inside one segment
        "root:photos/.%2e/secret.jpg",      // mixed-encoding dotdot
    ] {
        assert_eq!(
            expand_source_root(bad, &map),
            Err(ParseError::InvalidRoot(bad.to_string())),
            "{bad}"
        );
    }
    for (source, expanded) in [
        // Decoding happens exactly once, like every URL
        // layer: a double-encoded dot segment expands
        // to a literal `%2e%2e` file name under the
        // root, not to a dot segment anywhere.
        (
            "root:photos/%252e%252e/secret.jpg",
            "https://cdn.example.com/photos/%252e%252e/secret.jpg",
        ),
        // Three dots are a file name, not a dot segment.
        (
            "root:photos/%2e%2e./secret.jpg",
            "https://cdn.example.com/photos/%2e%2e./secret.jpg",
        ),
        // An `@`, `?`, or `#` inside the path is path
        // text: the base already fixed the scheme, host,
        // and prefix, so none of them can re-target the
        // fetch.
        (
            "root:photos/@evil.example/cat.jpg",
            "https://cdn.example.com/photos/@evil.example/cat.jpg",
        ),
        (
            "root:photos/cat.jpg?x=1",
            "https://cdn.example.com/photos/cat.jpg?x=1",
        ),
        (
            "root:photos/cat.jpg#frag",
            "https://cdn.example.com/photos/cat.jpg#frag",
        ),
        (
            "root:photos//evil.example/cat.jpg",
            "https://cdn.example.com/photos//evil.example/cat.jpg",
        ),
        // Trailing-dot segments are file names, not
        // dot segments.
        (
            "root:photos/cat.jpg.",
            "https://cdn.example.com/photos/cat.jpg.",
        ),
        (
            "root:photos/a./cat.jpg",
            "https://cdn.example.com/photos/a./cat.jpg",
        ),
        // A `..` that stays inside the root is a normal
        // path, however it is spelled.
        (
            "root:photos/a/b/../../cat.jpg",
            "https://cdn.example.com/photos/cat.jpg",
        ),
    ] {
        let (got, _) = expand_source_root(source, &map).unwrap();
        assert_eq!(got, expanded, "{source}");
        assert!(
            got.starts_with("https://cdn.example.com/photos/"),
            "{source}"
        );
    }
}

/// Issue #250: depth does not change the rules, and no
/// input in the adversarial space expands outside its
/// root's base or fails with anything but the two
/// frozen root codes.
#[test]
fn root_expansion_holds_its_invariants_at_any_depth() {
    let map = root_map();
    // Ten thousand segments deep, served fine...
    let deep = format!("root:photos/{}/cat.jpg", "a/".repeat(9_999));
    let (expanded, _) = expand_source_root(&deep, &map).unwrap();
    assert!(expanded.starts_with("https://cdn.example.com/photos/a/"));
    assert!(expanded.ends_with("/cat.jpg"));
    // ...and ten thousand climbs are refused by the
    // second one already.
    let climbs = format!("root:photos/a/{}/cat.jpg", "../".repeat(9_999));
    assert_eq!(
        expand_source_root(&climbs, &map),
        Err(ParseError::InvalidRoot(climbs.clone()))
    );
    // The sweep: every combination of an adversarial
    // name and an adversarial path either refuses with
    // a frozen root code or expands under the base of
    // the root it named.
    let names = ["photos", "media", "nope", "Photos", "", "🦀", "a b"];
    let paths = [
        "cat.jpg",
        "../cat.jpg",
        "%2e%2e/cat.jpg",
        "..%5ccat.jpg",
        "%252e%252e/cat.jpg",
        "@evil.example/cat.jpg",
        "a//b.jpg",
        ".",
        "..",
        "%",
        "a/./b/../cat.jpg",
        "cat.jpg?x",
        "cat.jpg#x",
    ];
    for name in names {
        for path in paths {
            let source = format!("root:{name}/{path}");
            match expand_source_root(&source, &map) {
                Ok((expanded, reference)) => {
                    let reference = reference.expect("a root source names its root");
                    let base = &map[&reference.name];
                    assert!(
                        expanded.starts_with(&format!("{base}/")),
                        "{source} expanded to {expanded}"
                    );
                }
                Err(e) => assert!(
                    matches!(e.code(), "unknown_root" | "invalid_root"),
                    "{source}: {e}"
                ),
            }
        }
    }
}

/// Issue #250: `root:` only has meaning as the very
/// start of the source. Inside a full source it is path
/// text, and a remainder whose source position holds a
/// malformed short form fails the same way the serve
/// path's own expansion fails.
#[test]
fn root_prefix_only_counts_at_the_source_start() {
    let map = root_map();
    assert_eq!(
        expand_source_root_in_remainder("300x200/https://example.com/root:photos/cat.jpg", &map)
            .unwrap(),
        "300x200/https://example.com/root:photos/cat.jpg"
    );
    assert_eq!(
        expand_source_root_in_remainder("300x200/root:photos", &map),
        Err(ParseError::InvalidRoot("root:photos".to_string()))
    );
    assert_eq!(
        expand_source_root_in_remainder("300x200/root:/cat.jpg", &map),
        Err(ParseError::InvalidRoot("root:/cat.jpg".to_string()))
    );
    assert_eq!(
        expand_source_root_in_remainder("300x200/root:photos/%2e%2e/secret.jpg", &map),
        Err(ParseError::InvalidRoot(
            "root:photos/%2e%2e/secret.jpg".to_string()
        ))
    );
}

#[test]
fn presets_only_mode_rejects_free_form_operations() {
    let map = preset_map();
    assert!(expand_presets("preset:thumb/example.com/img.jpg", &map, true).is_ok());
    assert!(expand_presets("exp:999/preset:thumb/example.com/img.jpg", &map, true).is_ok());
    assert!(expand_presets("maxage:60/preset:thumb/example.com/img.jpg", &map, true).is_ok());
    assert!(expand_presets("example.com/img.jpg", &map, true).is_ok());
    assert_eq!(
        expand_presets("300x200/example.com/img.jpg", &map, true),
        Err(ParseError::PresetsOnly("300x200".to_string()))
    );
    assert_eq!(
        expand_presets(
            "preset:thumb/filters:grayscale/example.com/i.jpg",
            &map,
            true
        ),
        Err(ParseError::PresetsOnly("filters:grayscale".to_string()))
    );
}

#[test]
fn preset_names_dedupe_in_first_seen_order() {
    assert_eq!(
        preset_names("preset:a/preset:b/preset:a/exp:999/preset:c/preset:b/example.com/i.jpg"),
        vec!["a", "b", "c"]
    );
    // The scan stops where expansion stops: a `preset:`
    // segment in the source path is URL text.
    assert_eq!(
        preset_names("preset:a/example.com/preset:b/i.jpg"),
        vec!["a"]
    );
    assert!(preset_names("example.com/i.jpg").is_empty());
    assert!(preset_names("preset:/example.com/i.jpg").is_empty());
}

#[test]
fn preset_names_stay_linear_on_long_distinct_lists() {
    // Hostile input (issue #409): the scan runs before
    // expansion rejects unknown names, so a remainder of
    // thousands of distinct names must cost linear work.
    let remainder = |names: &[String]| {
        let mut text: String = names.iter().map(|n| format!("preset:{n}/")).collect();
        text.push_str("example.com/img.jpg");
        text
    };
    // The report's shape: about 6,000 distinct short
    // names (a 65 KB remainder), then the same names
    // again, which fold into the first-seen order.
    let names: Vec<String> = (0..6_000).map(|i| format!("{i:x}")).collect();
    let doubled: Vec<String> = names.iter().chain(names.iter()).cloned().collect();
    assert_eq!(preset_names(&remainder(&names)), names);
    assert_eq!(preset_names(&remainder(&doubled)), names);
    // Linear, not quadratic: 50,000 distinct names take
    // milliseconds with the hash set, where the old scan
    // of the kept list made about 1.25 billion string
    // comparisons (seconds even in a release build).
    let many: Vec<String> = (0..50_000).map(|i| format!("{i:x}")).collect();
    let text = remainder(&many);
    let started = std::time::Instant::now();
    assert_eq!(preset_names(&text).len(), many.len());
    assert!(
        started.elapsed() < std::time::Duration::from_secs(2),
        "50,000 names took {:?}",
        started.elapsed()
    );
}

#[test]
fn parses_filter_batch_two() {
    let cases: Vec<(&str, Filter)> = vec![
        ("sharpen(1.5)", Filter::Sharpen(1.5)),
        ("saturation(0)", Filter::Saturation(0.0)),
        ("saturation(2.5)", Filter::Saturation(2.5)),
        ("rgb(255,-255,0)", Filter::Rgb(255, -255, 0)),
        ("equalize", Filter::Equalize),
        ("rotate(90)", Filter::Rotate(90.0)),
        ("rotate(-90)", Filter::Rotate(-90.0)),
        ("rotate(45)", Filter::Rotate(45.0)),
        ("rotate(45.5)", Filter::Rotate(45.5)),
        ("rotate(-30.25)", Filter::Rotate(-30.25)),
        ("fill(ff8800)", Filter::Fill([255, 136, 0])),
        ("max_bytes(5000)", Filter::MaxBytes(5000)),
    ];
    for (seg, want) in cases {
        let path = format!("filters:{seg}/example.com/img.jpg");
        let (ops, _) = parse_operations(&path).unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![want])], "{seg}");
    }
}

#[test]
fn default_ops_parse_and_expand() {
    let mut presets = std::collections::HashMap::new();
    presets.insert(
        "thumb".to_string(),
        "200x200/filters:format(webp)".parse().unwrap(),
    );
    // Plain operations.
    let ops = parse_default_ops("filters:quality(80)", &presets).unwrap();
    assert_eq!(ops.len(), 1);
    // Preset references expand at parse (boot) time.
    let ops = parse_default_ops("preset:thumb", &presets).unwrap();
    assert_eq!(ops.len(), 2);
    // Empty means none.
    assert!(parse_default_ops("", &presets).unwrap().is_empty());
    assert!(parse_default_ops("  ", &presets).unwrap().is_empty());
    // Unknown presets and bad operations are errors.
    assert!(parse_default_ops("preset:missing", &presets).is_err());
    assert!(parse_default_ops("filters:blur(0)", &presets).is_err());
    assert!(parse_default_ops("nonsense!!", &presets).is_err());
}

#[test]
fn default_ops_merge_drops_overridden_singletons() {
    let presets = std::collections::HashMap::new();
    let defaults = parse_default_ops("filters:format(webp):quality(80):blur(2)", &presets).unwrap();
    // A bare request keeps every default.
    let (bare, _) = parse_operations("example.com/i.jpg").unwrap();
    let merged = merge_default_ops(&defaults, bare);
    assert_eq!(
        canonical_form(&merged, "example.com/i.jpg"),
        "filters:format(webp):quality(80):blur(2)/example.com/i.jpg"
    );
    // A request that spells out format and quality drops
    // those defaults; the surviving default (blur) runs
    // first.
    let (explicit, _) =
        parse_operations("filters:format(png):quality(60)/example.com/i.jpg").unwrap();
    let merged = merge_default_ops(&defaults, explicit);
    assert_eq!(
        canonical_form(&merged, "example.com/i.jpg"),
        "filters:blur(2):format(png):quality(60)/example.com/i.jpg"
    );
    // Non-singleton defaults stack with the request's own
    // filters instead of being dropped.
    let (blurred, _) = parse_operations("filters:blur(3)/example.com/i.jpg").unwrap();
    let merged = merge_default_ops(&defaults, blurred);
    assert_eq!(
        canonical_form(&merged, "example.com/i.jpg"),
        "filters:format(webp):quality(80):blur(2):blur(3)/example.com/i.jpg"
    );
}

#[test]
fn default_ops_refuse_request_metadata() {
    // Issue #469: the parse used to accept `exp:` and
    // `maxage:` and drop them, so a default lifetime
    // booted cleanly and never applied. Both are refused
    // now, naming the segment, wherever they sit.
    let mut presets = std::collections::HashMap::new();
    presets.insert("thumb".to_string(), "100x0".parse().unwrap());
    for (text, want) in [
        (
            "filters:format(webp)/maxage:60",
            ParseError::InvalidMaxAge("maxage:60".to_string()),
        ),
        (
            "maxage:0",
            ParseError::InvalidMaxAge("maxage:0".to_string()),
        ),
        (
            "exp:1893456000/filters:format(webp)",
            ParseError::InvalidExpiry("exp:1893456000".to_string()),
        ),
        (
            "preset:thumb/exp:1/maxage:5",
            ParseError::InvalidExpiry("exp:1".to_string()),
        ),
        (
            "maxage:60/preset:thumb",
            ParseError::InvalidMaxAge("maxage:60".to_string()),
        ),
    ] {
        assert_eq!(parse_default_ops(text, &presets), Err(want), "{text}");
    }
    // Malformed metadata still fails as the parser's own
    // error, and metadata-free strings still parse.
    assert_eq!(
        parse_default_ops("maxage:soon", &presets),
        Err(ParseError::InvalidMaxAge("maxage:soon".to_string()))
    );
    assert_eq!(
        parse_default_ops("preset:thumb/filters:format(webp)", &presets)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn operation_count_counts_each_filter_and_every_other_operation_once() {
    let (ops, _) = parse_operations(
        "crop:1200x630:smart/filters:grayscale:quality(80)/exp:9/maxage:5/300x200/example.com/i.jpg",
    )
    .unwrap();
    // crop (1) + two filters (2) + size (1); exp and
    // maxage are not operations.
    assert_eq!(operation_count(&ops), 4);
    assert_eq!(operation_count(&[]), 0);
    assert!(check_operation_count(&[]).is_ok());
    // Exactly at the cap passes, either way it is spelled.
    let one_segment = vec![Operation::Filters(vec![Filter::Invert; MAX_OPERATIONS])];
    assert!(check_operation_count(&one_segment).is_ok());
    let segments = vec![Operation::Filters(vec![Filter::Invert]); MAX_OPERATIONS];
    assert!(check_operation_count(&segments).is_ok());
    // One more fails with the count in the message and the
    // engine's code, so the HTTP answer did not change when
    // the check moved to parse time.
    let mut over = segments.clone();
    over.push(Operation::Filters(vec![Filter::Invert]));
    let err = check_operation_count(&over).unwrap_err();
    assert_eq!(err, ParseError::TooManyOperations(MAX_OPERATIONS + 1));
    assert_eq!(err.code(), "unsupported_operation");
    assert_eq!(
        err.to_string(),
        "request has 101 operations, over the 100 operation limit"
    );
    // An empty filters list contributes nothing, and a
    // chain about as long as a 64 KiB URL can spell in one
    // segment is refused with its full count.
    assert_eq!(operation_count(&[Operation::Filters(Vec::new())]), 0);
    let (hostile, _) = parse_operations(&format!(
        "filters:{}/example.com/i.jpg",
        vec!["invert"; 9_000].join(":")
    ))
    .unwrap();
    assert_eq!(
        check_operation_count(&hostile),
        Err(ParseError::TooManyOperations(9_000))
    );
}

#[test]
fn default_ops_over_the_cap_alone_are_refused() {
    // Issue #469: defaults that alone exceed the cap would
    // make every request answer 400, so the boot refuses
    // them. The count is the expanded one: a preset counts
    // as the operations it expands to.
    let mut presets = std::collections::HashMap::new();
    presets.insert(
        "ten".to_string(),
        format!("filters:{}", ["invert"; 10].join(":"))
            .parse()
            .unwrap(),
    );
    let at_cap = format!("filters:{}", vec!["invert"; MAX_OPERATIONS].join(":"));
    assert_eq!(
        operation_count(&parse_default_ops(&at_cap, &presets).unwrap()),
        MAX_OPERATIONS
    );
    let over = format!("{at_cap}:invert");
    assert_eq!(
        parse_default_ops(&over, &presets),
        Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
    );
    let tens = ["preset:ten"; 10].join("/");
    assert!(parse_default_ops(&tens, &presets).is_ok());
    assert_eq!(
        parse_default_ops(&format!("{tens}/300x200"), &presets),
        Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
    );
}

#[test]
fn merged_count_drops_overridden_default_singletons() {
    // The cap counts the merged list: a default singleton
    // the URL overrides is dropped before counting, the
    // rest add to the URL's own operations.
    let presets = std::collections::HashMap::new();
    let defaults = parse_default_ops("filters:format(webp):blur(2)", &presets).unwrap();
    let ninety_eight = vec!["filters:invert"; MAX_OPERATIONS - 2].join("/");
    let (ops, _) = parse_operations(&format!("{ninety_eight}/example.com/i.jpg")).unwrap();
    assert!(check_operation_count(&merge_default_ops(&defaults, ops)).is_ok());
    let (ops, _) =
        parse_operations(&format!("{ninety_eight}/filters:invert/example.com/i.jpg")).unwrap();
    assert_eq!(
        check_operation_count(&merge_default_ops(&defaults, ops)),
        Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
    );
    // The URL's own format drops the default one, so 98
    // inverts plus a URL format still fit: the surviving
    // default blur, 98 inverts, and the format make 100,
    // where counting the unmerged lists would make 101.
    let (ops, _) = parse_operations(&format!(
        "{ninety_eight}/filters:format(png)/example.com/i.jpg"
    ))
    .unwrap();
    assert!(check_operation_count(&merge_default_ops(&defaults, ops)).is_ok());
}

#[test]
fn rotate_angles_canonicalize() {
    // Whole spellings keep the historic integer form (and
    // its cache key); fractions use the shortest form.
    assert_eq!(
        canonical_key_form("filters:rotate(90.0)/example.com/img.jpg"),
        "filters:rotate(90)/example.com/img.jpg"
    );
    assert_eq!(
        canonical_key_form("filters:rotate(45.50)/example.com/img.jpg"),
        "filters:rotate(45.5)/example.com/img.jpg"
    );
    assert_eq!(
        canonical_key_form("filters:rotate(-90)/example.com/img.jpg"),
        "filters:rotate(-90)/example.com/img.jpg"
    );
}

#[test]
fn rejects_bad_filter_batch_two() {
    for seg in [
        "sharpen(0)",
        "sharpen(-1)",
        "saturation(-0.5)",
        "rgb(256,0,0)",
        "rgb(1,2)",
        "equalize(1)",
        "rotate(1e2)",
        "rotate(.5)",
        "rotate(45.)",
        "rotate(nan)",
        "rotate(36001)",
        "rotate(-)",
        "fill(ff88)",
        "fill(gggggg)",
        "max_bytes(0)",
        "max_bytes(-5)",
    ] {
        let path = format!("filters:{seg}/example.com/img.jpg");
        assert!(parse_operations(&path).is_err(), "{seg}");
    }
}

#[test]
fn parses_new_filters() {
    let (ops, _) = parse_operations(
        "filters:invert():blur(2.5):brighten(10):contrast(1.2):huerotate(90)/example.com/img.jpg",
    )
    .unwrap();
    assert_eq!(
        ops,
        vec![Operation::Filters(vec![
            Filter::Invert,
            Filter::Blur(2.5),
            Filter::Brighten(10),
            Filter::Contrast(1.2),
            Filter::HueRotate(90),
        ])]
    );
}

#[test]
fn rejects_bad_new_filter_params() {
    for seg in [
        "filters:blur()",
        "filters:blur(0)",
        "filters:blur(-1.5)",
        "filters:blur(lots)",
        "filters:blur(513)",
        "filters:blur(512.5)",
        "filters:blur(1000000000)",
        "filters:blur(1e-45)",
        "filters:blur(1e-40)",
        "filters:blur(inf)",
        "filters:blur(NaN)",
        "filters:brighten()",
        "filters:brighten(lots)",
        "filters:contrast()",
        "filters:contrast(lots)",
        "filters:huerotate()",
        "filters:huerotate(lots)",
        "filters:invert(1)",
    ] {
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidFilters(seg.to_string())),
            "segment: {seg:?}"
        );
    }
}

#[test]
fn blur_sigma_cap_boundary() {
    // The cap itself, and a small normal sigma, are accepted.
    let (ops, _) = parse_operations("filters:blur(512)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Blur(512.0)])]);
    let (ops, _) = parse_operations("filters:blur(0.001)/example.com/img.jpg").unwrap();
    assert_eq!(ops, vec![Operation::Filters(vec![Filter::Blur(0.001)])]);
    // Just over the cap is invalid_filters, like any other
    // out-of-range filter parameter.
    assert_eq!(
        parse_operations("filters:blur(512.5)/example.com/img.jpg"),
        Err(ParseError::InvalidFilters(
            "filters:blur(512.5)".to_string()
        ))
    );
}

#[test]
fn keeps_operations_in_path_order() {
    let (ops, url) =
        parse_operations("crop:100x100/300x200/filters:grayscale()/example.com/x.jpg").unwrap();
    assert_eq!(ops.len(), 3);
    assert!(matches!(ops[0], Operation::Crop { .. }));
    assert!(matches!(ops[1], Operation::Size(_)));
    assert!(matches!(ops[2], Operation::Filters(_)));
    assert_eq!(url, "example.com/x.jpg");
}

#[test]
fn keeps_source_url_with_slashes_intact() {
    let (ops, url) = parse_operations("300x200/https://example.com/a/b/c.jpg").unwrap();
    assert_eq!(ops, vec![size_op(300, 200)]);
    assert_eq!(url, "https://example.com/a/b/c.jpg");
}

#[test]
fn rejects_missing_source_url() {
    for input in ["", "300x200", "crop:100x100/filters:grayscale()"] {
        assert_eq!(
            parse_operations(input),
            Err(ParseError::MissingSourceUrl),
            "input: {input:?}"
        );
    }
}

#[test]
fn non_operation_segment_starts_source_url() {
    // A segment is an operation only if it parses as one or carries a
    // known prefix; otherwise it starts the source URL. (A segment
    // shaped like a size is the one exception: it errors instead,
    // see malformed_size_segments_are_parse_errors_not_sources.)
    let (ops, url) = parse_operations("photos/img.jpg").unwrap();
    assert!(ops.is_empty());
    assert_eq!(url, "photos/img.jpg");
}

#[test]
fn parse_request_composes_signature_and_operations() {
    let (sig, _kid, req) =
        parse_request("/unsafe/300x200/filters:grayscale()/example.com/img.jpg").unwrap();
    assert_eq!(sig, Signature::Unsafe);
    assert_eq!(req.operations.len(), 2);
    assert!(matches!(req.operations[0], Operation::Size(_)));
    assert!(matches!(req.operations[1], Operation::Filters(_)));
    assert_eq!(req.source_url, "example.com/img.jpg");
}

#[test]
fn parse_errors_render() {
    assert_eq!(
        ParseError::InvalidSize("x".to_string()).to_string(),
        "invalid size segment: x"
    );
    assert_eq!(
        ParseError::MissingSourceUrl.to_string(),
        "missing source URL"
    );
}

#[test]
fn operation_queries_find_directives() {
    let (ops, _) =
        parse_operations("100x100/filters:format(png):no_upscale/example.com/i.jpg").unwrap();
    assert_eq!(
        RequestDirectives::analyze(&ops).explicit_format,
        Some(ImageFormat::Png)
    );
    assert!(RequestDirectives::analyze(&ops).no_upscale);
    let (ops, _) = parse_operations("100x100/example.com/i.jpg").unwrap();
    assert_eq!(RequestDirectives::analyze(&ops).explicit_format, None);
    assert!(!RequestDirectives::analyze(&ops).no_upscale);
    // A later format filter wins over an earlier one.
    let (ops, _) = parse_operations("filters:format(jpeg):format(webp)/example.com/i.jpg").unwrap();
    assert_eq!(
        RequestDirectives::analyze(&ops).explicit_format,
        Some(ImageFormat::WebP)
    );
}
