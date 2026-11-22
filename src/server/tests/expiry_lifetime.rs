//! The cache lifetime of an expiring URL's image responses.
//! A URL with an `exp:` segment answers with a `max-age` no
//! longer than the seconds left before that second, and no
//! longer than its `maxage:`, so a cache that follows the
//! response headers stops reusing a copy when the server
//! starts refusing the URL. Every answer that carries a
//! lifetime agrees: a render, a hit from each tier, a `304`,
//! a `HEAD`, a hint-shaped private answer, and the answers a
//! flight shares. A URL without `exp:`, or with one more than
//! a year away, keeps the immutable year.

use super::*;

/// How far ahead the URLs below expire, in seconds.
const LEFT: u64 = 100;

/// The `max-age` of a `Cache-Control` value.
fn max_age(cache_control: &str) -> Option<u64> {
    cache_control
        .split(',')
        .find_map(|d| d.trim().strip_prefix("max-age="))
        .and_then(|v| v.parse().ok())
}

/// Send `request`, returning its answer and the clock just
/// before it went out and just after its answer came back.
async fn timed(request: reqwest::RequestBuilder) -> (reqwest::Response, (u64, u64)) {
    let sent = now_unix();
    let resp = request.send().await.unwrap();
    (resp, (sent, now_unix()))
}

/// Assert that `cache_control` is `<visibility>, max-age=<n>,
/// immutable` for `n` the seconds left before `exp` at some
/// second between `sent` and `arrived`.
fn assert_ends_at(
    cache_control: &str,
    visibility: &str,
    exp: u64,
    (sent, arrived): (u64, u64),
    what: &str,
) {
    let age =
        max_age(cache_control).unwrap_or_else(|| panic!("{what}: no max-age in {cache_control:?}"));
    let (fewest, most) = (exp - arrived, exp - sent);
    assert!(
        (fewest..=most).contains(&age),
        "{what}: max-age {age} with {fewest} to {most} seconds left: {cache_control:?}"
    );
    assert_eq!(
        cache_control,
        format!("{visibility}, max-age={age}, immutable"),
        "{what}"
    );
}

/// The signed URL of `remainder` on `base`.
fn signed(base: &str, remainder: &str) -> String {
    format!("{base}/{}/{remainder}", sign(remainder))
}

#[tokio::test]
async fn an_expiring_urls_lifetime_ends_at_its_expiry_on_a_render_a_hit_and_a_304() {
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(false).await;
    let exp = now_unix() + LEFT;
    // Neither `exp:` nor `maxage:` is part of the cache key:
    // the first request renders, and every later one, with
    // whatever lifetime segments and in either spelling, is a
    // hit on that one entry.
    let expiring = format!("exp:{exp}/300x50/{src}");
    let (render, clock) = timed(client().get(signed(&base, &expiring))).await;
    assert_eq!(render.status(), StatusCode::OK);
    assert_eq!(header(&render, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_ends_at(
        &header(&render, "cache-control"),
        "public",
        exp,
        clock,
        "the render",
    );
    let etag = header(&render, "etag");
    // `None` for a lifetime that ends at the expiry.
    let cases = [
        ("exp", signed(&base, &expiring), None),
        (
            "exp, sealed",
            format!("{base}/{}", sealed_path(&expiring, Some("photo.png"))),
            None,
        ),
        (
            "exp and a shorter maxage",
            signed(&base, &format!("exp:{exp}/maxage:60/300x50/{src}")),
            Some("public, max-age=60, immutable"),
        ),
        (
            "exp and a longer maxage",
            signed(&base, &format!("maxage:1000/exp:{exp}/300x50/{src}")),
            None,
        ),
        (
            "exp and maxage:0",
            signed(&base, &format!("exp:{exp}/maxage:0/300x50/{src}")),
            Some("public, max-age=0"),
        ),
        (
            "no exp",
            signed(&base, &format!("300x50/{src}")),
            Some(YEAR_PUBLIC),
        ),
    ];
    for (label, url, fixed) in cases {
        for (pass, request, status) in [
            ("hit", client().get(&url), StatusCode::OK),
            (
                "304",
                client().get(&url).header("If-None-Match", &etag),
                StatusCode::NOT_MODIFIED,
            ),
            ("HEAD", client().head(&url), StatusCode::OK),
        ] {
            let what = format!("{label} ({pass})");
            let (resp, clock) = timed(request).await;
            assert_eq!(resp.status(), status, "{what}");
            assert_eq!(header(&resp, "cache-status"), "crabcrop; hit", "{what}");
            let cache_control = header(&resp, "cache-control");
            match fixed {
                Some(value) => assert_eq!(cache_control, value, "{what}"),
                None => assert_ends_at(&cache_control, "public", exp, clock, &what),
            }
        }
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1, "one render answered all");
}

#[tokio::test]
async fn a_disk_hit_carries_the_lifetime_an_expiry_leaves() {
    let dir = crate::test_support::scratch_dir("srv-exp-disk");
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    // Memory tier off: every repeat comes from disk.
    let base = spawn_app_full(std::collections::HashMap::new(), Some(&dir), 0).await;
    let exp = now_unix() + LEFT;
    let url = signed(&base, &format!("exp:{exp}/40x24/{src}"));
    let (render, clock) = timed(client().get(&url)).await;
    assert_eq!(header(&render, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_ends_at(
        &header(&render, "cache-control"),
        "public",
        exp,
        clock,
        "the render",
    );
    let etag = header(&render, "etag");
    for (pass, request, status) in [
        ("hit", client().get(&url), StatusCode::OK),
        (
            "304",
            client().get(&url).header("If-None-Match", &etag),
            StatusCode::NOT_MODIFIED,
        ),
    ] {
        let (resp, clock) = timed(request).await;
        assert_eq!(resp.status(), status, "{pass}");
        assert_eq!(
            header(&resp, "cache-status"),
            "crabcrop; hit; detail=\"disk\"",
            "{pass}"
        );
        assert_ends_at(&header(&resp, "cache-control"), "public", exp, clock, pass);
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_shared_hit_carries_the_lifetime_an_expiry_leaves() {
    let fake = crate::s3store::test_support::start_fake().await;
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let exp = now_unix() + LEFT;
    let remainder = format!("exp:{exp}/25x25/{src}");
    let path = format!("{}/{remainder}", sign(&remainder));
    // One replica renders, and writes the render through to
    // the bucket in the background.
    let writer = spawn_app_with_shared(None, &fake.endpoint).await;
    let (render, clock) = timed(client().get(format!("{writer}/{path}"))).await;
    assert_eq!(render.status(), StatusCode::OK);
    assert_ends_at(
        &header(&render, "cache-control"),
        "public",
        exp,
        clock,
        "the render",
    );
    let etag = header(&render, "etag");
    let mut landed = false;
    for _ in 0..50 {
        if entry_objects(&fake) > 0 {
            landed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(landed, "the shared write-through never landed");
    // Each answer below comes from a fresh replica, its
    // memory cold and with no disk, so the bucket answers.
    for (pass, conditional, status) in [
        ("hit", false, StatusCode::OK),
        ("304", true, StatusCode::NOT_MODIFIED),
    ] {
        let reader = spawn_app_with_shared(None, &fake.endpoint).await;
        let mut request = client().get(format!("{reader}/{path}"));
        if conditional {
            request = request.header("If-None-Match", &etag);
        }
        let (resp, clock) = timed(request).await;
        assert_eq!(resp.status(), status, "{pass}");
        let cache_status = header(&resp, "cache-status");
        assert!(
            cache_status.contains("detail=\"shared\""),
            "{pass}: {cache_status}"
        );
        assert_ends_at(&header(&resp, "cache-control"), "public", exp, clock, pass);
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_hint_shaped_answer_keeps_the_lifetime_an_expiry_leaves() {
    // A DPR hint makes the answer private (see
    // `keep_from_shared_caches`): the lifetime it keeps is
    // the one the expiry left.
    let (src, _) = spawn_counting_server(200, "image/png", png_of(640, 480)).await;
    let base = spawn_app(false).await;
    let exp = now_unix() + LEFT;
    let url = signed(
        &base,
        &format!("exp:{exp}/300x0/filters:format(jpeg)/{src}"),
    );
    let (render, clock) = timed(client().get(&url).header("DPR", "2")).await;
    assert_eq!(header(&render, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_ends_at(
        &header(&render, "cache-control"),
        "private",
        exp,
        clock,
        "the render",
    );
    let etag = header(&render, "etag");
    for (pass, request, status) in [
        ("hit", client().get(&url).header("DPR", "2"), StatusCode::OK),
        (
            "304",
            client()
                .get(&url)
                .header("DPR", "2")
                .header("If-None-Match", &etag),
            StatusCode::NOT_MODIFIED,
        ),
    ] {
        let (resp, clock) = timed(request).await;
        assert_eq!(resp.status(), status, "{pass}");
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit", "{pass}");
        assert_ends_at(&header(&resp, "cache-control"), "private", exp, clock, pass);
    }
}

#[tokio::test]
async fn a_flights_shared_answers_keep_the_lifetime_an_expiry_leaves() {
    // With every cache tier off, a render is stored nowhere,
    // and a flight's waiters answer with its leader's
    // response (issue #407): its lifetime still ends at the
    // expiry, on the 200s and the 304s alike.
    let (src, _, _) = spawn_slow_peak_server(Some(test_png()), 300).await;
    let base = spawn_app_full(std::collections::HashMap::new(), None, 0).await;
    let exp = now_unix() + LEFT;
    let url = signed(&base, &format!("exp:{exp}/50x30/{src}"));
    let (solo, clock) = timed(client().get(&url)).await;
    assert_eq!(solo.status(), StatusCode::OK);
    assert_ends_at(
        &header(&solo, "cache-control"),
        "public",
        exp,
        clock,
        "the render",
    );
    let etag = header(&solo, "etag");
    let sent = now_unix();
    let tasks: Vec<_> = (0..8)
        .map(|i| {
            let mut request = client().get(&url);
            if i % 2 == 0 {
                request = request.header("If-None-Match", &etag);
            }
            tokio::spawn(async move {
                let resp = request.send().await.unwrap();
                (resp.status(), header(&resp, "cache-control"))
            })
        })
        .collect();
    let mut answers = Vec::new();
    for task in tasks {
        answers.push(task.await.unwrap());
    }
    let arrived = now_unix();
    for (i, (status, cache_control)) in answers.iter().enumerate() {
        let expected = if i % 2 == 0 {
            StatusCode::NOT_MODIFIED
        } else {
            StatusCode::OK
        };
        assert_eq!(*status, expected, "request {i}");
        assert_ends_at(
            cache_control,
            "public",
            exp,
            (sent, arrived),
            &format!("request {i}"),
        );
    }
}

#[tokio::test]
async fn a_preset_urls_expiry_sets_its_lifetime() {
    // A remainder that names a preset is parsed on the
    // expanding path, which reads the expiry too.
    let (src, _) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_presets(preset_test_map(), false).await;
    let exp = now_unix() + LEFT;
    let url = signed(&base, &format!("exp:{exp}/preset:small/{src}"));
    for pass in ["fwd=uri-miss", "hit"] {
        let (resp, clock) = timed(client().get(&url)).await;
        assert_eq!(resp.status(), StatusCode::OK, "{pass}");
        assert_eq!(header(&resp, "cache-status"), format!("crabcrop; {pass}"));
        assert_ends_at(&header(&resp, "cache-control"), "public", exp, clock, pass);
    }
}

#[tokio::test]
async fn a_render_that_ends_past_the_expiry_is_stale_on_arrival() {
    // The seconds left are counted as the response is
    // composed, not when the request was admitted: this
    // request is admitted two or three seconds before its
    // expiry, and its source takes 3.5 seconds, so its render
    // ends past the expiry second and leaves with no lifetime.
    let (src, _) = spawn_slow_counting_server(test_png(), 3500).await;
    let base = spawn_app(false).await;
    let exp = now_unix() + 3;
    let url = signed(&base, &format!("exp:{exp}/300x50/{src}"));
    let resp = client().get(&url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-control"), "public, max-age=0");
    // A revalidation from here on is refused, as the
    // response's lifetime said.
    let etag = header(&resp, "etag");
    let resp = client()
        .get(&url)
        .header("If-None-Match", &etag)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[test]
fn an_image_lifetime_is_the_shorter_of_its_maxage_and_the_seconds_its_expiry_leaves() {
    let cache_control = |max_age: Option<u32>, expires_at: Option<u64>| {
        let resp = respond(
            ImageFormat::Png,
            None,
            bytes::Bytes::from_static(b"png"),
            max_age,
            expires_at,
            true,
        );
        resp.headers()[header::CACHE_CONTROL]
            .to_str()
            .unwrap()
            .to_string()
    };
    let now = now_unix();
    let year = u64::from(crate::url::MAX_CACHE_AGE);
    // No expiry, or one more than a year away: the header a
    // URL without `exp:` gets, byte for byte.
    for expires_at in [None, Some(now + year + 60), Some(u64::MAX)] {
        assert_eq!(cache_control(None, expires_at), YEAR_PUBLIC);
        assert_eq!(
            cache_control(Some(60), expires_at),
            "public, max-age=60, immutable"
        );
        assert_eq!(cache_control(Some(0), expires_at), "public, max-age=0");
    }
    // At or past the expiry second: stale on arrival, with
    // no `immutable`, whatever the `maxage:` says.
    for expires_at in [0, 1, now - 1, now] {
        for max_age in [None, Some(0), Some(60)] {
            assert_eq!(
                cache_control(max_age, Some(expires_at)),
                "public, max-age=0",
                "{max_age:?} at {expires_at}"
            );
        }
    }
    // Before it: the seconds left, or the `maxage:` when that
    // is shorter.
    let exp = now + LEFT;
    let sent = now_unix();
    let alone = cache_control(None, Some(exp));
    let longer = cache_control(Some(1000), Some(exp));
    let arrived = now_unix();
    assert_ends_at(&alone, "public", exp, (sent, arrived), "exp alone");
    assert_ends_at(&longer, "public", exp, (sent, arrived), "a longer maxage");
    assert_eq!(
        cache_control(Some(60), Some(exp)),
        "public, max-age=60, immutable"
    );
}

#[tokio::test]
async fn an_expiry_more_than_a_year_away_keeps_the_immutable_year() {
    // The lifetime is capped at the expiry, never raised to
    // it: past a year away, the answer is the one a URL
    // without `exp:` gets, byte for byte.
    let (src, _) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(false).await;
    let far = now_unix() + 2 * u64::from(crate::url::MAX_CACHE_AGE);
    for (remainder, expected) in [
        (format!("exp:{far}/300x50/{src}"), YEAR_PUBLIC),
        (
            format!("exp:{far}/maxage:60/300x50/{src}"),
            "public, max-age=60, immutable",
        ),
        (
            format!("exp:{far}/maxage:0/300x50/{src}"),
            "public, max-age=0",
        ),
    ] {
        for pass in ["first", "again"] {
            let resp = get_signed(&base, &remainder).await;
            assert_eq!(resp.status(), StatusCode::OK, "{remainder} ({pass})");
            assert_eq!(
                header(&resp, "cache-control"),
                expected,
                "{remainder} ({pass})"
            );
        }
    }
}

#[tokio::test]
async fn an_expired_url_is_refused_on_every_path() {
    // From its expiry second on, a URL is refused before the
    // tiers are read, so none of its answers carries a
    // lifetime: not with its render cached under the plain
    // spelling, not for a revalidation of that render, and
    // not for a HEAD.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(false).await;
    let cached = get_signed(&base, &format!("300x50/{src}")).await;
    assert_eq!(cached.status(), StatusCode::OK);
    let etag = header(&cached, "etag");
    let now = now_unix();
    for exp in [now, now - 1, 1] {
        let url = signed(&base, &format!("exp:{exp}/300x50/{src}"));
        for (pass, request) in [
            ("GET", client().get(&url)),
            (
                "conditional GET",
                client().get(&url).header("If-None-Match", &etag),
            ),
            ("HEAD", client().head(&url)),
        ] {
            let what = format!("exp:{exp} ({pass})");
            let resp = request.send().await.unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{what}");
            assert_eq!(header(&resp, "cache-control"), "no-store", "{what}");
            assert!(header(&resp, "etag").is_empty(), "{what}");
            if pass != "HEAD" {
                assert_eq!(resp.text().await.unwrap(), "signature expired", "{what}");
            }
        }
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}
