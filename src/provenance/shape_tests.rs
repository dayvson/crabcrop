//! Every shape of manifest the signer builds (issue #669).
//! A request's signing no longer has the SDK validate the
//! manifest it has just signed, unless a timestamp authority
//! is configured, so each shape is proven here to read back
//! valid, and the boot probe is proven to sign each one with
//! that validation on and to fail on one assembled wrongly.

use super::*;

/// The six shapes [`ManifestParts::of`] builds, as (has a
/// parent ingredient, the ingredient records a status, the
/// request ran an AI edit).
const SHAPES: [(bool, bool, bool); 6] = [
    (false, false, false),
    (false, false, true),
    (true, false, false),
    (true, false, true),
    (true, true, false),
    (true, true, true),
];

/// The shape of `parts`, as [`SHAPES`] lists them.
fn shape_of(parts: &ManifestParts) -> (bool, bool, bool) {
    (
        parts.ingredient.is_some(),
        parts.records_status(),
        parts.actions.len() == 2,
    )
}

/// The self-test image signed with the test identity: a
/// source whose own manifest store validates.
fn source_with_a_store(signer: &ProvenanceSigner) -> Vec<u8> {
    signer
        .sign(
            SELF_TEST_PNG,
            "image/png",
            &SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .expect("the source signs")
}

/// The request that asks for `shape`, over `stored` as the
/// source that carries a manifest store.
fn request_for((parent, status, ai_edited): (bool, bool, bool), stored: &[u8]) -> SignRequest<'_> {
    let source = if status { stored } else { SELF_TEST_PNG };
    SignRequest {
        ingredient: parent.then_some((source, "image/png")),
        ai_edited,
    }
}

/// A reader with the SDK's own defaults, validation
/// included.
fn validating_reader(signed: &[u8], mime: &str) -> Reader {
    Reader::from_context(Context::new())
        .with_stream(mime, Cursor::new(signed))
        .expect("the reader parses the output")
}

/// The active manifest's actions as the reader reports them:
/// each one's name and digital source type.
fn actions(reader: &Reader) -> Vec<(String, Option<String>)> {
    let store: serde_json::Value = serde_json::from_str(&reader.json()).expect("the store is JSON");
    let label = store["active_manifest"]
        .as_str()
        .expect("an active manifest");
    let assertions = store["manifests"][label]["assertions"]
        .as_array()
        .expect("the manifest lists its assertions");
    let listed = assertions
        .iter()
        .find(|assertion| {
            assertion["label"]
                .as_str()
                .is_some_and(|label| label.starts_with("c2pa.actions"))
        })
        .expect("an actions assertion");
    listed["data"]["actions"]
        .as_array()
        .expect("the assertion lists its actions")
        .iter()
        .map(|action| {
            (
                action["action"].as_str().expect("a name").to_string(),
                action["digitalSourceType"].as_str().map(str::to_string),
            )
        })
        .collect()
}

/// Assert that `signed` reads back valid and holds exactly
/// what `shape` says.
fn assert_shape(signed: &[u8], mime: &str, shape: (bool, bool, bool)) {
    let (parent, status, ai_edited) = shape;
    let what = format!("{shape:?} as {mime}");
    let reader = validating_reader(signed, mime);
    assert_eq!(reader.validation_state(), ValidationState::Valid, "{what}");
    let summary = read_manifest(signed, mime).expect("the output carries a manifest");
    assert_eq!(summary.state, "valid", "{what}");
    assert_eq!(summary.manifests, 1, "{what}");
    assert_eq!(summary.ingredients, usize::from(parent), "{what}");
    let manifest = reader.active_manifest().expect("an active manifest");
    if let [ingredient] = manifest.ingredients() {
        assert!(ingredient.is_parent(), "{what}");
        assert_eq!(ingredient.format(), Some("image/png"), "{what}");
        assert_eq!(ingredient.validation_results().is_some(), status, "{what}");
    }
    let mut expected = vec![if parent {
        ("c2pa.opened".to_string(), None)
    } else {
        ("c2pa.created".to_string(), Some(SOFTWARE_IMAGE.to_string()))
    }];
    if ai_edited {
        expected.push((
            "c2pa.edited".to_string(),
            Some(AI_EDITED_SOURCE_TYPE.to_string()),
        ));
    }
    assert_eq!(actions(&reader), expected, "{what}");
}

#[test]
fn the_six_shapes_are_every_shape_a_request_can_ask_for() {
    // A request has two fields and a source either carries a
    // manifest store or does not: every combination, and
    // nothing else, is one of the six.
    let signer = tests::test_signer();
    let stored = source_with_a_store(&signer);
    let mut built = Vec::new();
    for source in [None, Some(SELF_TEST_PNG), Some(&stored[..])] {
        for ai_edited in [false, true] {
            built.push(shape_of(&ManifestParts::of(&SignRequest {
                ingredient: source.map(|bytes| (bytes, "image/png")),
                ai_edited,
            })));
        }
    }
    built.sort();
    assert_eq!(built, SHAPES);
    for shape in SHAPES {
        assert_eq!(
            shape_of(&ManifestParts::of(&request_for(shape, &stored))),
            shape
        );
    }
}

#[test]
fn every_shape_signed_as_a_request_signs_it_reads_back_valid() {
    let signer = tests::test_signer();
    // No timestamp authority: the context a request signs
    // under leaves the SDK's own validation off.
    assert!(!signer.signer.context().settings().verify.verify_after_sign);
    let stored = source_with_a_store(&signer);
    for shape in SHAPES {
        let signed = signer
            .sign(SELF_TEST_PNG, "image/png", &request_for(shape, &stored))
            .unwrap_or_else(|e| panic!("{shape:?}: {e}"));
        assert_shape(&signed, "image/png", shape);
    }
}

#[test]
fn every_signed_format_reads_back_valid() {
    // The four formats the server signs, each with the
    // barest manifest and the fullest.
    let pixels = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(16, 16, |x, y| {
        image::Rgb([(x * 16) as u8, (y * 16) as u8, 90])
    }));
    let encoded = |format| {
        let mut out = Cursor::new(Vec::new());
        pixels
            .write_to(&mut out, format)
            .expect("the output encodes");
        out.into_inner()
    };
    let signer = tests::test_signer();
    let stored = source_with_a_store(&signer);
    for (mime, output) in [
        ("image/jpeg", encoded(image::ImageFormat::Jpeg)),
        ("image/png", encoded(image::ImageFormat::Png)),
        ("image/webp", encoded(image::ImageFormat::WebP)),
        ("image/avif", encoded(image::ImageFormat::Avif)),
    ] {
        for shape in [SHAPES[0], SHAPES[5]] {
            let signed = signer
                .sign(&output, mime, &request_for(shape, &stored))
                .unwrap_or_else(|e| panic!("{shape:?} as {mime}: {e}"));
            assert_shape(&signed, mime, shape);
        }
    }
}

/// The IPTC source type of an output rendered by software.
const SOFTWARE_IMAGE: &str = "http://cv.iptc.org/newscodes/digitalsourcetype/softwareImage";

/// `parts`, assembled wrongly: a second action that opens
/// the manifest, where the specification allows only one.
fn assembled_wrongly(mut parts: ManifestParts) -> ManifestParts {
    parts.actions.push(serde_json::json!({
        "action": "c2pa.created",
        "digitalSourceType": SOFTWARE_IMAGE,
    }));
    parts
}

#[test]
fn the_validation_a_request_skips_is_the_one_that_refuses_a_wrong_manifest() {
    let signer = tests::test_signer();
    let stored = source_with_a_store(&signer);
    // Each shape with a second opening action, and the two
    // mistakes the comments in `ManifestParts::of` name: a
    // created action with no digital source type, and an
    // opened action that names no ingredient.
    let mut wrong: Vec<(String, Box<dyn Fn() -> ManifestParts + '_>)> = Vec::new();
    for shape in SHAPES {
        let stored = &stored;
        wrong.push((
            format!("{shape:?} with two opening actions"),
            Box::new(move || assembled_wrongly(ManifestParts::of(&request_for(shape, stored)))),
        ));
    }
    wrong.push((
        "created with no source type".to_string(),
        Box::new(|| {
            let mut parts = ManifestParts::of(&request_for(SHAPES[0], &stored));
            parts.actions[0] = serde_json::json!({ "action": "c2pa.created" });
            parts
        }),
    ));
    wrong.push((
        "opened with no ingredient".to_string(),
        Box::new(|| {
            let mut parts = ManifestParts::of(&request_for(SHAPES[2], &stored));
            parts.actions[0] = serde_json::json!({ "action": "c2pa.opened" });
            parts
        }),
    ));
    for (what, parts) in &wrong {
        // With the validation on, the SDK refuses to hand
        // the manifest back.
        let refused = signer.sign_parts(
            signer.signer.validating_context(),
            SELF_TEST_PNG,
            "image/png",
            parts(),
        );
        assert!(
            matches!(refused, Err(SignError::Failed(_))),
            "{what}: {:?}",
            refused.map(|signed| signed.len())
        );
        // Under the request's context it is signed as it
        // is, and reads back invalid: this is what the boot
        // probe is there to catch, for every shape.
        let signed = signer
            .sign_parts(signer.signer.context(), SELF_TEST_PNG, "image/png", parts())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let summary = read_manifest(&signed, "image/png").expect("the output carries a manifest");
        assert_eq!(summary.state, "invalid", "{what}");
    }
}

#[test]
fn with_a_timestamp_authority_a_request_keeps_the_validation() {
    // Building a signer contacts nobody: the authority is
    // only asked when something is signed.
    let authority = TimestampAuthority::new("http://127.0.0.1:9/".to_string());
    let timestamped = ProvenanceSigner::build(
        TEST_CHAIN.as_bytes(),
        TEST_KEY.as_bytes(),
        SigningAlg::Es256,
        Some(authority),
    )
    .expect("the test identity loads");
    let context = timestamped.signer.context();
    assert!(context.settings().verify.verify_after_sign);
    // The boot probe's context validates with or without
    // one.
    let plain = tests::test_signer();
    for signer in [&timestamped, &plain] {
        let context = signer.signer.validating_context();
        assert!(context.settings().verify.verify_after_sign);
    }
}

/// Load the test identity as boot does, with the shape
/// probe's manifests built by `parts`.
fn boot_with(
    parts: impl Fn(&SignRequest<'_>) -> ManifestParts,
) -> Result<ProvenanceSigner, String> {
    ProvenanceSigner::from_pem_probing(
        TEST_CHAIN.as_bytes().to_vec(),
        TEST_KEY.as_bytes().to_vec(),
        None,
        parts,
    )
}

#[test]
fn boot_signs_every_shape_with_validation_on() {
    // Loading the identity asks for each of the six shapes
    // once, after the identity's own probe has passed.
    let asked = std::cell::RefCell::new(Vec::new());
    let signer = boot_with(|request| {
        let parts = ManifestParts::of(request);
        asked.borrow_mut().push(shape_of(&parts));
        parts
    })
    .expect("the test identity loads");
    assert_eq!(signer.name, "crabcrop test signer");
    let mut asked = asked.into_inner();
    asked.sort();
    assert_eq!(asked, SHAPES);
    // The probe alone, as boot runs it.
    assert_eq!(signer.probe_shapes(ManifestParts::of), Ok(()));
}

#[test]
fn boot_fails_on_any_one_shape_assembled_wrongly() {
    for (shape, named) in SHAPES.into_iter().zip([
        "a manifest without an ingredient:",
        "a manifest without an ingredient, AI-edited:",
        "a manifest with a parent ingredient:",
        "a manifest with a parent ingredient, AI-edited:",
        "a manifest with a parent ingredient and its recorded status:",
        "a manifest with a parent ingredient and its recorded status, AI-edited:",
    ]) {
        let err = boot_with(|request| {
            let parts = ManifestParts::of(request);
            if shape_of(&parts) == shape {
                assembled_wrongly(parts)
            } else {
                parts
            }
        })
        .expect_err("a wrong manifest fails the boot");
        assert!(
            err.starts_with("the signing self-test failed on ") && err.contains(named),
            "{shape:?}: {err}"
        );
    }
    // A parent whose status goes unrecorded is a shape the
    // probe did not sign, and it says so.
    let err = boot_with(|request| {
        let mut parts = ManifestParts::of(request);
        if let Some(ingredient) = parts.ingredient.as_mut().and_then(|i| i.as_object_mut()) {
            ingredient.remove("validation_results");
        }
        parts
    })
    .expect_err("an unrecorded status fails the boot");
    assert!(
        err.contains("the parent's validation status was not recorded"),
        "{err}"
    );
}

/// `text` with every UUID in it replaced by zeros.
fn without_uuids(text: &str) -> String {
    let bytes = text.as_bytes();
    let is_uuid = |window: &[u8]| {
        window.iter().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => *b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
    };
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes.len() - at >= 36 && is_uuid(&bytes[at..at + 36]) {
            out.push_str("00000000-0000-0000-0000-000000000000");
            at += 36;
        } else {
            // A UUID is ASCII, so it never starts inside a
            // character of several bytes.
            let end = (at + 1..=bytes.len())
                .find(|&end| text.is_char_boundary(end))
                .expect("the text ends on a boundary");
            out.push_str(&text[at..end]);
            at = end;
        }
    }
    out
}

/// A manifest report with what is fresh in every signature
/// set aside: the UUIDs a manifest names itself and its
/// output by, and the hashes of the assertions that hold
/// them.
fn without_identifiers(report: &mut serde_json::Value) {
    match report {
        serde_json::Value::String(text) => *text = without_uuids(text),
        serde_json::Value::Array(items) => items.iter_mut().for_each(without_identifiers),
        serde_json::Value::Object(fields) => {
            let renamed: serde_json::Map<String, serde_json::Value> = std::mem::take(fields)
                .into_iter()
                .map(|(key, mut value)| {
                    if key == "hash" {
                        value = serde_json::Value::Null;
                    } else {
                        without_identifiers(&mut value);
                    }
                    (without_uuids(&key), value)
                })
                .collect();
            *fields = renamed;
        }
        _ => {}
    }
}

#[test]
fn skipping_the_validation_changes_nothing_that_is_written() {
    // The same request signed under the boot probe's context
    // (validation on) and under a request's (validation off):
    // an output of the same length, whose manifest reports
    // the same thing once its fresh identifiers are set
    // aside.
    let signer = tests::test_signer();
    let stored = source_with_a_store(&signer);
    for shape in SHAPES {
        let request = request_for(shape, &stored);
        let sign_under = |context: Context| {
            signer
                .sign_parts(
                    context,
                    SELF_TEST_PNG,
                    "image/png",
                    ManifestParts::of(&request),
                )
                .unwrap_or_else(|e| panic!("{shape:?}: {e}"))
        };
        let validated = sign_under(signer.signer.validating_context());
        let unvalidated = sign_under(signer.signer.context());
        assert_eq!(validated.len(), unvalidated.len(), "{shape:?}");
        let report = |signed: &[u8]| {
            let mut report: serde_json::Value =
                serde_json::from_str(&validating_reader(signed, "image/png").json())
                    .expect("the report is JSON");
            without_identifiers(&mut report);
            report
        };
        let (with, without) = (report(&validated), report(&unvalidated));
        assert_eq!(with, without, "{shape:?}");
        // The report is not an empty one: it names the
        // signer, and the parent where there is one.
        let text = with.to_string();
        assert!(text.contains("crabcrop test signer"), "{shape:?}: {text}");
        assert_eq!(text.contains("c2pa.ingredient"), shape.0, "{shape:?}");
        // Two signatures of one request differ, so the
        // comparison above is not of one output with itself.
        assert!(validated != unvalidated, "{shape:?}");
    }
    assert_eq!(
        without_uuids("urn:c2pa:3fa85f64-5717-4562-b3fc-2c963f66afa6:x"),
        "urn:c2pa:00000000-0000-0000-0000-000000000000:x"
    );
}
