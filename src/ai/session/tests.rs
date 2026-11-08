use super::*;

#[test]
fn ort_search_path_prefers_the_ort_dylib_variable() {
    // #180: the fallback resolution mirrors ort's own, so
    // the preflight loads exactly what ort would have
    // loaded lazily (and panicked on).
    assert_eq!(
        ort_search_path(Some("/opt/ort/libonnxruntime.so")),
        std::path::PathBuf::from("/opt/ort/libonnxruntime.so")
    );
    let default = std::path::PathBuf::from(format!(
        "{}onnxruntime{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    assert_eq!(ort_search_path(None), default);
    assert_eq!(ort_search_path(Some("")), default);
}
