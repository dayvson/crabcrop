use super::*;
use image::GrayImage;

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

#[test]
fn fir_resize_runs_rgb_and_gray_through_one_kernel() {
    // One body serves both layouts (issue #484): a gray
    // image resampled as RGB and as Luma lands on the same
    // value in every channel, at the requested size, up,
    // down, and at its own size.
    let gray = GrayImage::from_fn(37, 23, |x, y| image::Luma([((x * 7 + y * 11) % 256) as u8]));
    let rgb = image::DynamicImage::ImageLuma8(gray.clone()).to_rgb8();
    for (w, h) in [(64, 40), (13, 7), (37, 23)] {
        let resized_gray = fir_resize(&gray, w, h);
        let resized_rgb = fir_resize(&rgb, w, h);
        assert_eq!(resized_gray.dimensions(), (w, h));
        assert_eq!(resized_rgb.dimensions(), (w, h));
        for (g, c) in resized_gray.pixels().zip(resized_rgb.pixels()) {
            assert_eq!(c.0, [g.0[0]; 3], "{w}x{h}");
        }
    }
}

/// The threads of this process: a pool's workers are threads
/// of the process that holds the session.
#[cfg(target_os = "linux")]
fn thread_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("this process's thread list")
        .count()
}

/// The helper of
/// `every_model_runs_on_the_runtimes_one_thread_pool`, run in
/// a process of its own so that no other test's threads are
/// counted. It starts the runtime, then builds three sessions
/// of one model, and prints the process's thread count after
/// the runtime and after each session. Without the variable it
/// is a test that does nothing.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "run by every_model_runs_on_the_runtimes_one_thread_pool"]
fn thread_pool_helper() {
    let Ok(model) = std::env::var("CRABCROP_TEST_POOL_HELPER_MODEL") else {
        return;
    };
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    environment(Some(&dylib)).unwrap();
    let mut counts = vec![thread_count()];
    let mut sessions = Vec::new();
    for _ in 0..3 {
        sessions.push(build_session(Some(&dylib), &model).unwrap());
        counts.push(thread_count());
    }
    let counts: Vec<String> = counts.iter().map(usize::to_string).collect();
    println!("pool_helper_threads={}", counts.join(","));
}

/// Every session runs on the runtime's one thread pool (issue
/// #693): a session built after the runtime started adds no
/// thread. Run manually with the deployment artifacts in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
/// cargo test -- --ignored every_model_runs_on_the_runtimes_one_thread_pool
#[cfg(target_os = "linux")]
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn every_model_runs_on_the_runtimes_one_thread_pool() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_IDENTITY_MODEL").unwrap();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "ai::session::tests::thread_pool_helper",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("CRABCROP_TEST_ORT_DYLIB", &dylib)
        .env("CRABCROP_TEST_POOL_HELPER_MODEL", &model)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the helper failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let counts: Vec<usize> = stdout
        .split("pool_helper_threads=")
        .nth(1)
        .and_then(|rest| rest.lines().next())
        .unwrap_or_else(|| panic!("the helper printed no thread counts:\n{stdout}"))
        .split(',')
        .map(|count| count.trim().parse().unwrap())
        .collect();
    // With a pool for each session, each of the three would
    // add a thread per core but one.
    assert_eq!(
        counts[1..],
        [counts[0]; 3],
        "threads after the runtime started, then after each of three sessions"
    );
}
