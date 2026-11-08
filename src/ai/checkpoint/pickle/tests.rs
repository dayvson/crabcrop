use super::*;

#[test]
fn unpickle_refuses_an_unknown_callable_by_name() {
    // A reduce of os.system is the canonical
    // hostile pickle; the machine refuses it
    // before anything runs (nothing here can run:
    // the machine only builds data).
    let pkl = b"\x80\x02cos\nsystem\n)R.".to_vec();
    let err = unpickle(&pkl).unwrap_err();
    assert!(
        err.contains("unsupported pickle callable os.system"),
        "{err}"
    );
}
