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
