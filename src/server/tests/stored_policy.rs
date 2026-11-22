//! Issue #780: a stored result answers only a request the
//! source policy still admits. A render is stored under the
//! policy it was made under, and the policy is part of no
//! cache key, so once the policy narrows (a host dropped from
//! the allowlist, the private-address check applied to listed
//! hosts, a bucket dropped from its list, file sources turned
//! off), a tier that holds a render of a source it now refuses
//! must not answer with it. Each test renders under the wider
//! policy and asks again under the narrower one: the request is
//! answered as a render of the refused source is, and the
//! stored copy is left where it is.

use super::*;
use crate::loader::{SourceConfig, SourcePolicy};

/// A loader that fetches from the hosts `listed` names only,
/// with the private-address check on for any other host, as
/// `CRABCROP_ALLOWED_SOURCES` sets it up.
fn listing(listed: &[&str]) -> Loader {
    Loader::with_defaults()
        .unwrap()
        .with_policy(policy_listing(listed))
}

/// The source policy [`listing`] loads under.
fn policy_listing(listed: &[&str]) -> SourcePolicy {
    SourcePolicy::new(listed.iter().map(|h| h.to_string()).collect(), true)
}

/// A state over the disk tier at `dir`, loading through
/// `loader`: a server that starts over the directory a
/// previous one left.
fn disk_state(dir: &std::path::Path, loader: Loader) -> Arc<AppState> {
    Arc::new(AppState {
        store: Some(test_store(dir)),
        loader,
        ..test_state()
    })
}

/// The key a plain request for `remainder` is stored under.
fn key_of(remainder: &str) -> String {
    let (ops, source, _, _) = parse_operations_with_expiry(remainder).unwrap();
    cache_key_for("test", &ops, source, &plain_negotiation(&ops))
}

/// What a request for `remainder` answers: its status, the
/// headers that tell a refusal from a hit, and its body.
async fn answer(base: &str, remainder: &str) -> (StatusCode, [String; 3], Vec<u8>) {
    let resp = get_signed(base, remainder).await;
    let headers = ["content-type", "cache-control", "cache-status"].map(|name| header(&resp, name));
    (resp.status(), headers, resp.bytes().await.unwrap().to_vec())
}

/// That `stored`, whose render a tier of the server at `base`
/// holds, is refused exactly as `unstored`, a render of the
/// same source that no tier holds: with the policy's `403`.
async fn assert_refused_as_a_render(base: &str, stored: &str, unstored: &str) {
    let refused = answer(base, stored).await;
    assert_eq!(
        refused.0,
        StatusCode::FORBIDDEN,
        "{stored} answered {:?} {:?}",
        refused.0,
        refused.1
    );
    assert_eq!(refused.2, b"source refused by policy", "{stored}");
    assert_eq!(refused, answer(base, unstored).await, "{stored}");
}

#[tokio::test]
async fn a_render_from_a_host_dropped_from_the_allowlist_never_reaches_memory() {
    // Memory is the server's own, and a memory hit runs no
    // check: a copy enters it only once this server's policy has
    // admitted its sources. Restarted under the narrower policy
    // over a disk tier and a bucket that both hold the render,
    // the server refuses it on every request, reads neither tier
    // for it, and so never promotes it into memory.
    let fake = crate::s3store::test_support::start_fake().await;
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let (stored, unstored) = (format!("40x24/{src}"), format!("41x24/{src}"));
    let dir = crate::test_support::scratch_dir("srv-policy-memory");
    let replica = |loader: Loader| {
        Arc::new(AppState {
            loader,
            ..tier_test_state(Some(&dir), &fake.endpoint)
        })
    };
    let wide = replica(listing(&["127.0.0.1"]));
    let resp = get_signed(&serve_state(&wide).await, &stored).await;
    assert_eq!(resp.status(), StatusCode::OK);
    writes_settle(&wide).await;
    assert_eq!(
        entry_objects(&fake),
        1,
        "the render was not written through"
    );
    let entry_reads = || {
        fake.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|line| line.starts_with("GET ") && line.ends_with(".bin"))
            .count()
    };
    let narrow = replica(listing(&["images.example.com"]));
    let key = key_of(&stored);
    assert!(narrow.store.as_ref().unwrap().contains(&key));
    let base = serve_state(&narrow).await;
    let reads = entry_reads();
    for _ in 0..2 {
        assert_refused_as_a_render(&base, &stored, &unstored).await;
        assert!(
            narrow.cache.get(&key).is_none(),
            "a refusal promoted the render"
        );
    }
    assert_eq!(entry_reads(), reads, "a refusal read the bucket copy");
    assert!(narrow.store.as_ref().unwrap().contains(&key));
    assert_eq!(entry_objects(&fake), 1);
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "a refusal fetched the source"
    );
    // Under a policy that still lists the host, the disk copy
    // serves and is promoted, and memory answers the next
    // request.
    let admitting = replica(listing(&["127.0.0.1", "images.example.com"]));
    let base = serve_state(&admitting).await;
    let resp = get_signed(&base, &stored).await;
    assert_eq!(header(&resp, "cache-status"), DISK_HIT);
    let resp = get_signed(&base, &stored).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_render_from_a_host_dropped_from_the_allowlist_is_refused_from_disk() {
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let (stored, unstored) = (format!("40x24/{src}"), format!("41x24/{src}"));
    let dir = crate::test_support::scratch_dir("srv-policy-disk");
    let wide = disk_state(&dir, listing(&["127.0.0.1"]));
    let resp = get_signed(&serve_state(&wide).await, &stored).await;
    assert_eq!(resp.status(), StatusCode::OK);
    // Restarted over the same directory with the source's host
    // off the allowlist.
    let narrow = disk_state(&dir, listing(&["images.example.com"]));
    let store = narrow.store.clone().unwrap();
    let key = key_of(&stored);
    assert!(store.contains(&key), "the restart did not index the render");
    assert_refused_as_a_render(&serve_state(&narrow).await, &stored, &unstored).await;
    // The refusal read no tier: it promoted nothing into memory
    // and left the disk copy.
    assert!(narrow.cache.get(&key).is_none());
    assert!(store.contains(&key));
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "a refusal fetched the source"
    );
    // Restarted again under a policy that still lists the host,
    // the copy serves from disk, with no fetch.
    let admitting = disk_state(&dir, listing(&["127.0.0.1", "images.example.com"]));
    let resp = get_signed(&serve_state(&admitting).await, &stored).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), DISK_HIT);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_render_from_a_host_dropped_from_the_allowlist_is_refused_from_the_shared_tier() {
    let fake = crate::s3store::test_support::start_fake().await;
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let (stored, unstored) = (format!("40x24/{src}"), format!("41x24/{src}"));
    let replica = |loader: Loader| {
        Arc::new(AppState {
            loader,
            ..tier_test_state(None, &fake.endpoint)
        })
    };
    let wide = replica(listing(&["127.0.0.1"]));
    let resp = get_signed(&serve_state(&wide).await, &stored).await;
    assert_eq!(resp.status(), StatusCode::OK);
    writes_settle(&wide).await;
    assert_eq!(
        entry_objects(&fake),
        1,
        "the render was not written through"
    );
    // Another replica, with the source's host off the
    // allowlist. Its refusal reads nothing from the bucket.
    let entry_reads = || {
        fake.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|line| line.starts_with("GET ") && line.ends_with(".bin"))
            .count()
    };
    let narrow = replica(listing(&["images.example.com"]));
    let reads = entry_reads();
    assert_refused_as_a_render(&serve_state(&narrow).await, &stored, &unstored).await;
    assert_eq!(entry_reads(), reads, "a refusal read the bucket copy");
    assert!(narrow.cache.get(&key_of(&stored)).is_none());
    assert_eq!(entry_objects(&fake), 1);
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "a refusal fetched the source"
    );
    // A replica whose policy still lists the host is served
    // the bucket copy.
    let admitting = replica(listing(&["127.0.0.1", "images.example.com"]));
    let resp = get_signed(&serve_state(&admitting).await, &stored).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        header(&resp, "cache-status"),
        "crabcrop; hit; detail=\"shared\""
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

/// One narrowing of the policy for the disk-tier test below:
/// what the loader loads through before and after, and a
/// render it then refuses.
struct Narrowing {
    what: &'static str,
    wide: Loader,
    narrow: Loader,
    stored: String,
    unstored: String,
}

#[tokio::test]
async fn every_narrowing_the_text_decides_refuses_a_stored_render() {
    // The allowlist is one narrowing. The private-address check
    // applied to listed hosts (`CRABCROP_CHECK_ALLOWED_SOURCES`),
    // a bucket dropped from `CRABCROP_ALLOWED_BUCKETS`,
    // `CRABCROP_FILE_ROOT` unset, and a watermark's host dropped
    // from the allowlist are the others: each refuses the render
    // stored under the wider policy, from the disk a restart
    // reads.
    let (src, _) = spawn_counting_server(200, "image/png", test_png()).await;
    let wm = spawn_watermark_server().await;
    let fake = crate::s3store::test_support::start_fake().await;
    fake.objects
        .lock()
        .unwrap()
        .insert("src.png".to_string(), test_png());
    let files = crate::test_support::scratch_dir("srv-policy-files");
    let root = std::fs::canonicalize(&files).unwrap();
    std::fs::write(root.join("src.png"), test_png()).unwrap();
    let buckets = |allowed: &str| SourceConfig {
        s3_endpoint: Some(fake.endpoint.clone()),
        aws_creds: Some(crate::s3store::test_support::test_aws()),
        allowed_buckets: vec![allowed.to_string()],
        ..Default::default()
    };
    let file_sources = || SourceConfig {
        file_root: Some(root.clone()),
        ..Default::default()
    };
    let watermarked = |size: &str| {
        let encoded = wm.replace(':', "%3A").replace('/', "%2F");
        format!("{size}/filters:watermark({encoded},0,0,100)/file:///src.png")
    };
    let cases = [
        Narrowing {
            what: "the private-address check on listed hosts",
            wide: listing(&["127.0.0.1"]),
            narrow: Loader::with_defaults()
                .unwrap()
                .with_policy(policy_listing(&["127.0.0.1"]).with_allowlisted_checked(true)),
            stored: format!("40x24/{src}"),
            unstored: format!("41x24/{src}"),
        },
        Narrowing {
            what: "a bucket dropped from the bucket list",
            wide: Loader::with_defaults()
                .unwrap()
                .with_sources(buckets("pics")),
            narrow: Loader::with_defaults()
                .unwrap()
                .with_sources(buckets("other")),
            stored: "40x24/s3://pics/src.png".to_string(),
            unstored: "41x24/s3://pics/src.png".to_string(),
        },
        Narrowing {
            what: "file sources turned off",
            wide: Loader::with_defaults()
                .unwrap()
                .with_sources(file_sources()),
            narrow: Loader::with_defaults().unwrap(),
            stored: "40x24/file:///src.png".to_string(),
            unstored: "41x24/file:///src.png".to_string(),
        },
        Narrowing {
            what: "a watermark's host dropped from the allowlist",
            wide: listing(&["127.0.0.1"]).with_sources(file_sources()),
            narrow: listing(&["images.example.com"]).with_sources(file_sources()),
            stored: watermarked("40x24"),
            unstored: watermarked("41x24"),
        },
    ];
    // Every case runs before the verdict, so a failure names
    // each narrowing that let a stored render serve.
    let mut served = Vec::new();
    for case in cases {
        let dir = crate::test_support::scratch_dir("srv-policy-narrowing");
        let wide = disk_state(&dir, case.wide);
        let resp = get_signed(&serve_state(&wide).await, &case.stored).await;
        assert_eq!(resp.status(), StatusCode::OK, "{}", case.what);
        let narrow = disk_state(&dir, case.narrow);
        let key = key_of(&case.stored);
        assert!(
            narrow.store.as_ref().unwrap().contains(&key),
            "{}: the restart did not index the render",
            case.what
        );
        let base = serve_state(&narrow).await;
        let (status, headers, _) = answer(&base, &case.stored).await;
        if status != StatusCode::FORBIDDEN {
            served.push(format!("{}: {status} {headers:?}", case.what));
            continue;
        }
        assert_refused_as_a_render(&base, &case.stored, &case.unstored).await;
        assert!(narrow.cache.get(&key).is_none(), "{}", case.what);
    }
    assert!(served.is_empty(), "stored renders served: {served:#?}");
}

#[tokio::test]
async fn a_hit_on_a_named_source_looks_nothing_up() {
    // The checks a hit runs read the URL's text alone: a name
    // the private-address check resolves on a miss is not
    // looked up on a hit. A render stored from a name that now
    // resolves to a non-public address therefore serves from
    // the tiers until it is purged; a render that is not stored
    // fails as such a name does.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let src = src.replace("127.0.0.1", "origin.test");
    let lookups = Arc::new(AtomicUsize::new(0));
    let inward = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (counted, turned) = (lookups.clone(), inward.clone());
    let loader = Loader::with_defaults()
        .unwrap()
        .with_policy(SourcePolicy::new(Vec::new(), true))
        .with_host_lookup(
            move |_| {
                counted.fetch_add(1, Ordering::SeqCst);
                let addr = if turned.load(Ordering::SeqCst) {
                    "10.0.0.8"
                } else {
                    "127.0.0.1"
                };
                Ok(vec![addr.parse().unwrap()])
            },
            crate::loader::LOOKUP_SLOTS,
            crate::loader::LOOKUP_WAIT,
        )
        .with_loopback_public();
    let stored = format!("40x24/{src}");
    let dir = crate::test_support::scratch_dir("srv-policy-lookup");
    let first = disk_state(&dir, loader.clone());
    let first_base = serve_state(&first).await;
    let resp = get_signed(&first_base, &stored).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(lookups.load(Ordering::SeqCst), 1);
    inward.store(true, Ordering::SeqCst);
    let resp = get_signed(&first_base, &stored).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    let restarted = disk_state(&dir, loader);
    let base = serve_state(&restarted).await;
    let resp = get_signed(&base, &stored).await;
    assert_eq!(header(&resp, "cache-status"), DISK_HIT);
    assert_eq!(
        lookups.load(Ordering::SeqCst),
        1,
        "a hit looked the name up"
    );
    let resp = get_signed(&base, &format!("41x24/{src}")).await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(lookups.load(Ordering::SeqCst), 2);
    // Purged, the render is a miss, which looks the name up and
    // fails like the render that was never stored.
    let resp = client()
        .post(format!(
            "{base}/purge/{}/{stored}",
            crate::url::purge_signature(SECRET, &stored)
        ))
        .send()
        .await
        .unwrap();
    // Its memory copy, promoted by the disk hit, and its disk
    // copy.
    assert_eq!(resp.text().await.unwrap(), "purged 2\n");
    let resp = get_signed(&base, &stored).await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(lookups.load(Ordering::SeqCst), 3);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}
