use super::*;

#[test]
fn ai_error_display_keeps_the_cause() {
    // The typed variants carry the same messages the
    // plain strings carried, so boot diagnostics and log
    // lines read as they did before AP-16.
    assert_eq!(
        AiError::Load("cannot load the model x: boom".to_string()).to_string(),
        "cannot load the model x: boom"
    );
    assert_eq!(
        AiError::Inference("inference failed: boom".to_string()).to_string(),
        "inference failed: boom"
    );
    assert_eq!(
        AiError::BadOutput("unexpected model output shape: [1]".to_string()).to_string(),
        "unexpected model output shape: [1]"
    );
    assert!(AiError::LockPoisoned.to_string().contains("poisoned"));
}

/// A lookup over a fixed set of variables, standing in for
/// the process environment in the [`provisioned`] tests.
fn lookup(
    set: &'static [(&'static str, &'static str)],
) -> impl Fn(&str) -> Result<String, std::env::VarError> {
    move |name| {
        set.iter()
            .find(|(var, _)| *var == name)
            .map(|(_, value)| value.to_string())
            .ok_or(std::env::VarError::NotPresent)
    }
}

#[test]
fn a_model_variable_that_is_set_is_used_as_given() {
    let set = lookup(&[("CRABCROP_TEST_FACE_MODEL", "/models/yunet.onnx")]);
    assert_eq!(
        provisioned("CRABCROP_TEST_FACE_MODEL", set).as_deref(),
        Some("/models/yunet.onnx")
    );
}

#[test]
fn an_unset_model_variable_skips_the_test() {
    assert_eq!(provisioned("CRABCROP_TEST_FACE_MODEL", lookup(&[])), None);
    // An empty requirement asks for nothing.
    let set = lookup(&[("CRABCROP_TEST_REQUIRE_MODELS", "")]);
    assert_eq!(provisioned("CRABCROP_TEST_FACE_MODEL", set), None);
}

/// Issue #710: the ai-models workflow provisions every model,
/// so there a missing variable is a test that silently did
/// not run, and it fails instead.
#[test]
#[should_panic(expected = "CRABCROP_TEST_FACE_MODEL is not set, and CRABCROP_TEST_REQUIRE_MODELS")]
fn an_unset_model_variable_fails_where_every_model_is_required() {
    let set = lookup(&[("CRABCROP_TEST_REQUIRE_MODELS", "1")]);
    provisioned("CRABCROP_TEST_FACE_MODEL", set);
}
