//! Which `/explain` probes take the probe gate. A JPEG XL
//! source's size comes from jxl-rs's basic-info event, which
//! the decoder reports only once it has read the header's
//! colour encoding, so explain sizes a JPEG XL source as it
//! sizes an SVG, an AVIF or a HEIC one: once, on a blocking
//! thread under the probe gate, never on the async runtime
//! after each chunk of the load.

use super::*;

/// A small JPEG XL file that sizes instantly but ends with the
/// probe's slow-probe marker
/// ([`crate::engine::decode::SLOW_PROBE_MARKER`]), so the
/// probe's test hook sleeps while sizing it: a stand-in for a
/// JPEG XL whose header is slow to read. The marker trails the
/// codestream, where the header read never reaches.
fn slow_probe_jxl() -> Vec<u8> {
    use crate::url::{Filter, FormatSpec, ImageFormat, Operation, PlainFilter};
    let format = Operation::Filters(vec![Filter::Plain(PlainFilter::Format(FormatSpec::plain(
        ImageFormat::Jxl,
    )))]);
    let mut jxl = crate::engine::Engine
        .process(&test_png(), &[format])
        .unwrap()
        .bytes;
    jxl.extend_from_slice(b"crabcrop-slow-probe");
    jxl
}

/// An explain app whose file root holds `name` with `bytes`,
/// and the root, which goes when the guard is dropped.
async fn explain_app_with_file(
    name: &str,
    bytes: &[u8],
) -> (String, crate::test_support::ScratchDir) {
    let dir = crate::test_support::scratch_dir("explain-probe-gate");
    let root = std::fs::canonicalize(&dir).unwrap();
    std::fs::write(root.join(name), bytes).unwrap();
    let loader = Loader::with_defaults()
        .unwrap()
        .with_sources(crate::loader::SourceConfig {
            file_root: Some(root),
            ..Default::default()
        });
    let base = spawn_explain_app_with_roots(Default::default(), loader, Default::default()).await;
    (base, dir)
}

/// The probe of a JPEG XL source is heavy, so the load never
/// sizes a JPEG XL head on the runtime after each chunk, and
/// the handler sizes it once off the runtime; the report still
/// carries the size.
#[tokio::test]
async fn explain_sizes_a_jxl_once_off_the_runtime() {
    let jxl = slow_probe_jxl();
    assert_eq!(crate::engine::probe_kind(&jxl), "jxl");
    assert!(crate::engine::probe_is_heavy("jxl"));
    assert!(!probe_settled(&jxl));
    let (base, dir) = explain_app_with_file("o.jxl", &jxl).await;
    let remainder = "64x64/file:///o.jxl";
    let resp = client()
        .get(format!("{base}/explain/{}/{remainder}", sign(remainder)))
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(json["source"]["kind"], "jxl");
    assert_eq!(json["source"]["width"], 100);
    assert_eq!(json["source"]["height"], 60);
    drop(dir);
}

/// A burst of /explain probes for a slow-to-size JPEG XL file
/// source leaves /health responsive, as one for an SVG does
/// (see `concurrent_slow_svg_file_explains_leave_health_responsive`):
/// the probes run on blocking threads under the probe gate. On
/// a server that sized JPEG XL on the request threads, the
/// probes would hold both of them and /health would wait for
/// one to finish.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_slow_jxl_file_explains_leave_health_responsive() {
    let (base, dir) = explain_app_with_file("slow.jxl", &slow_probe_jxl()).await;
    // Distinct output sizes, so the requests are distinct and
    // each runs its own probe.
    let urls = (0..6)
        .map(|i| {
            let remainder = format!("64x{}/file:///slow.jxl", 60 + i);
            format!("{base}/explain/{}/{remainder}", sign(&remainder))
        })
        .collect();
    health_stays_responsive_during_slow_explains(urls, &base).await;
    drop(dir);
}
