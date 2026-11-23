//! The `CDN-Loop` mark (RFC 8586). Every fetch the server
//! makes for a request (its source, each watermark and
//! overlay, a fallback image fetched in its place, the explain
//! report's probe) carries the server's own entry, after the
//! entries the request itself arrived with. An image, meta or
//! explain request that carries the server's entry is one of
//! those fetches come back to the server, and answers
//! `508 loop_detected` before its cache lookup and before any
//! fetch. A request marked by another deployment or another
//! CDN, and one with no mark at all, is served as before.

use super::*;

/// An origin that serves the test PNG on every connection and
/// keeps the head of every request it reads, in arrival order.
async fn spawn_recording_origin() -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    let heads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let kept = heads.clone();
    let png = Arc::new(test_png());
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let (kept, png) = (kept.clone(), png.clone());
            tokio::spawn(async move {
                let mut seen = Vec::new();
                let mut buf = [0u8; 4096];
                while !seen.windows(4).any(|w| w == b"\r\n\r\n") && seen.len() < 64 * 1024 {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => seen.extend_from_slice(&buf[..n]),
                    }
                }
                kept.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&seen).into_owned());
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    png.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(&png).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (format!("http://{addr}"), heads)
}

/// The `CDN-Loop` lines of one request head, in order.
fn cdn_loop_lines(head: &str) -> Vec<String> {
    head.lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("cdn-loop")
                .then(|| value.trim().to_string())
        })
        .collect()
}

/// The path a request head asks for.
fn path_of(head: &str) -> &str {
    head.split(' ').nth(1).unwrap_or_default()
}

/// The head `origin` kept for `path`, waiting a little for it.
async fn head_for(heads: &std::sync::Mutex<Vec<String>>, path: &str) -> String {
    for _ in 0..200 {
        if let Some(head) = heads
            .lock()
            .unwrap()
            .iter()
            .find(|head| path_of(head) == path)
        {
            return head.clone();
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("the origin never saw a request for {path}");
}

/// The one `CDN-Loop` line a fetch for `path` carried.
async fn mark_for(heads: &std::sync::Mutex<Vec<String>>, path: &str) -> String {
    let head = head_for(heads, path).await;
    let lines = cdn_loop_lines(&head);
    assert_eq!(lines.len(), 1, "one CDN-Loop line for {path}: {head}");
    lines[0].clone()
}

/// Whether `entry` is the server's own: `crabcrop; id=` and a
/// 32-digit lowercase hex token.
fn is_own_entry(entry: &str) -> bool {
    entry.strip_prefix("crabcrop; id=").is_some_and(|token| {
        token.len() == 32
            && token
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// `state` served on a free local port, with the handle the
/// test keeps on it.
async fn serve_shared(state: AppState) -> (String, Arc<AppState>) {
    let state = Arc::new(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = app(Arc::clone(&state));
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), state)
}

/// The count `/metrics` gives image requests of `class`.
fn requests_of(state: &AppState, class: &str) -> u64 {
    let rendered = state.metrics.render();
    let prefix = format!("crabcrop_requests_total{{status_class=\"{class}\"}} ");
    rendered
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .and_then(|count| count.trim().parse().ok())
        .unwrap_or_else(|| panic!("no {class} count in {rendered}"))
}

/// The server's own entry, as its fetch for a fresh render of
/// a source on `origin` carries it.
async fn own_entry(base: &str, origin: &str, heads: &std::sync::Mutex<Vec<String>>) -> String {
    let resp = client()
        .get(format!("{base}/unsafe/20x0/{origin}/learn.png"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let entry = mark_for(heads, "/learn.png").await;
    assert!(is_own_entry(&entry), "{entry:?}");
    entry
}

#[tokio::test]
async fn a_source_that_points_back_at_the_server_is_refused_at_its_first_nested_hop() {
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let (base, state) = serve_shared(AppState {
        allow_unsafe: true,
        ..test_state()
    })
    .await;
    // Three levels of the server's own URL before the origin's.
    let mut source = src;
    for _ in 0..3 {
        source = format!("{base}/unsafe/10x10/{source}");
    }
    let resp = client()
        .get(format!("{base}/unsafe/10x10/{source}"))
        .send()
        .await
        .unwrap();
    // The outer request answers what a failed source fetch
    // answers: its source, the first nested hop, answered 508.
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(header(&resp, "x-crabcrop-fallback"), "");
    assert_eq!(resp.text().await.unwrap(), "source returned status 508");
    // Two requests in all: the outer one and its first nested
    // hop. No deeper hop ran, and the origin was never asked.
    assert_eq!(requests_of(&state, "5xx"), 2);
    assert_eq!(requests_of(&state, "2xx"), 0);
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

/// An origin that answers every request with a redirect to
/// `location`.
async fn spawn_redirecting_origin(location: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let head = format!(
        "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            reply_and_close(&mut socket, head.as_bytes(), b"").await;
        }
    });
    format!("http://{addr}/moved.png")
}

#[tokio::test]
async fn a_redirect_back_to_the_server_is_refused_at_its_first_nested_hop() {
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let (base, state) = serve_shared(AppState {
        allow_unsafe: true,
        ..test_state()
    })
    .await;
    // The origin sends the fetch back to the server, whose
    // URL names a source of its own.
    let back = spawn_redirecting_origin(format!("{base}/unsafe/10x10/{src}")).await;
    let resp = client()
        .get(format!("{base}/unsafe/10x10/{back}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(resp.text().await.unwrap(), "source returned status 508");
    assert_eq!(requests_of(&state, "5xx"), 2);
    assert_eq!(requests_of(&state, "2xx"), 0);
    assert_eq!(hits.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn every_fetch_a_request_makes_carries_the_servers_entry() {
    let (origin, heads) = spawn_recording_origin().await;
    let (missing, _) = spawn_counting_server(404, "text/plain", b"gone".to_vec()).await;
    let (base, _state) = serve_shared(AppState {
        allow_unsafe: true,
        fallback: Some(Fallback::Url(format!("{origin}/fallback.png"))),
        ..test_state()
    })
    .await;
    let encode = |url: String| url.replace(':', "%3A").replace('/', "%2F");
    let wm = encode(format!("{origin}/wm.png"));
    let ov = encode(format!("{origin}/ov.png"));
    // A render with a watermark and an overlay: three fetches.
    let resp = client()
        .get(format!(
            "{base}/unsafe/50x30/filters:watermark({wm},0,0,100):overlay({ov},0,0,50,multiply)/{origin}/src.png"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // A source the origin does not have: the fallback image is
    // fetched in its place.
    let resp = client()
        .get(format!("{base}/unsafe/20x0/{missing}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "x-crabcrop-fallback"), "1");
    // The explain report's probe.
    let resp = client()
        .get(format!("{base}/explain/unsafe/20x0/{origin}/explain.png"))
        .header("accept", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let entry = mark_for(&heads, "/src.png").await;
    assert!(is_own_entry(&entry), "{entry:?}");
    for path in ["/wm.png", "/ov.png", "/fallback.png", "/explain.png"] {
        assert_eq!(mark_for(&heads, path).await, entry, "the fetch of {path}");
    }
}

#[tokio::test]
async fn a_request_carrying_the_servers_entry_answers_508_before_its_cache_lookup() {
    let (origin, heads) = spawn_recording_origin().await;
    let (base, state) = serve_shared(AppState {
        allow_unsafe: true,
        ..test_state()
    })
    .await;
    let entry = own_entry(&base, &origin, &heads).await;
    // The URL is rendered and cached first: a marked request
    // for it is refused all the same, from no tier.
    let url = format!("{base}/unsafe/30x0/{origin}/cached.png");
    let resp = client().get(&url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let fetched = heads.lock().unwrap().len();
    let explain = format!("{base}/explain/unsafe/30x0/{origin}/cached.png");
    let json = |request: reqwest::RequestBuilder| request.header("accept", "application/json");
    for (what, request) in [
        ("image", client().get(&url).header("cdn-loop", &entry)),
        ("HEAD", client().head(&url).header("cdn-loop", &entry)),
        ("meta", json(client().get(&url)).header("cdn-loop", &entry)),
        (
            "explain",
            json(client().get(&explain)).header("cdn-loop", &entry),
        ),
        (
            "an entry among others, on a second line",
            client()
                .get(&url)
                .header("cdn-loop", "cloudflare")
                .header("cdn-loop", format!("fastly, {entry}, akamai;v=1.0")),
        ),
    ] {
        let resp = request.send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::LOOP_DETECTED, "{what}");
        assert_eq!(header(&resp, "cache-control"), "no-store", "{what}");
        assert_eq!(header(&resp, "cache-status"), "", "{what}");
        if what == "meta" || what == "explain" {
            let body: serde_json::Value =
                serde_json::from_str(&resp.text().await.unwrap()).unwrap();
            assert_eq!(body["error"]["code"], "loop_detected", "{what}: {body}");
        }
    }
    // The bare entry, in the plain text body every client gets.
    let resp = client()
        .get(&url)
        .header("cdn-loop", &entry)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::LOOP_DETECTED);
    assert_eq!(
        resp.text().await.unwrap(),
        "loop detected: the request carries this server's CDN-Loop entry"
    );
    // No refusal fetched anything. Each image-route refusal
    // left its record, as every image request does.
    assert_eq!(heads.lock().unwrap().len(), fetched);
    assert_eq!(requests_of(&state, "5xx"), 5);
}

#[tokio::test]
async fn a_marked_request_is_refused_before_its_signature_is_read() {
    // Signed mode: each URL below is refused for its
    // signature or token when it carries no mark. Marked,
    // each answers 508 before either is read.
    let (origin, heads) = spawn_recording_origin().await;
    let (base, state) = serve_shared(test_state()).await;
    let entry = state.loader.loop_mark().entry().clone();
    for path in [
        format!("/unsafe/20x0/{origin}/a.png"),
        format!("/{}/20x0/{origin}/a.png", "0".repeat(64)),
        "/s1.AAAA/a.png".to_string(),
        format!("/explain/unsafe/20x0/{origin}/a.png"),
    ] {
        let plain = client().get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(plain.status(), StatusCode::FORBIDDEN, "{path}");
        let marked = client()
            .get(format!("{base}{path}"))
            .header("cdn-loop", entry.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(marked.status(), StatusCode::LOOP_DETECTED, "{path}");
    }
    assert!(heads.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_request_marked_by_another_deployment_or_cdn_is_served() {
    let (origin, _heads) = spawn_recording_origin().await;
    let base = spawn_app(true).await;
    for (what, marks) in [
        (
            "another deployment",
            vec!["crabcrop; id=0123456789abcdef0123456789abcdef"],
        ),
        ("another CDN", vec!["cloudflare"]),
        (
            "both, over two lines",
            vec![
                "Fastly, crabcrop; id=ffffffffffffffffffffffffffffffff",
                "akamai;v=1.0",
            ],
        ),
        ("a bare crabcrop entry", vec!["crabcrop"]),
    ] {
        let mut request = client().get(format!("{base}/unsafe/20x0/{origin}/other.png"));
        for mark in marks {
            request = request.header("cdn-loop", mark);
        }
        let resp = request.send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{what}");
        assert_eq!(header(&resp, "content-type"), "image/png", "{what}");
    }
}

#[tokio::test]
async fn a_request_without_the_header_is_answered_as_before() {
    let (origin, heads) = spawn_recording_origin().await;
    let base = spawn_app(true).await;
    let url = format!("{base}/unsafe/20x0/{origin}/plain.png");
    let render = client().get(&url).send().await.unwrap();
    assert_eq!(render.status(), StatusCode::OK);
    assert_eq!(header(&render, "content-type"), "image/png");
    assert_eq!(header(&render, "cache-control"), YEAR_PUBLIC);
    assert_eq!(header(&render, "cache-status"), "crabcrop; fwd=uri-miss");
    let bytes = render.bytes().await.unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (20, 12));
    let hit = client().get(&url).send().await.unwrap();
    assert_eq!(hit.status(), StatusCode::OK);
    assert_eq!(header(&hit, "cache-status"), "crabcrop; hit");
    assert_eq!(hit.bytes().await.unwrap(), bytes);
    // One fetch, for the render.
    assert_eq!(heads.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_fetch_carries_the_requests_own_entries_ahead_of_the_servers() {
    let (origin, heads) = spawn_recording_origin().await;
    let base = spawn_app(true).await;
    let entry = own_entry(&base, &origin, &heads).await;
    let other = "crabcrop; id=0123456789abcdef0123456789abcdef";
    let resp = client()
        .get(format!("{base}/unsafe/20x0/{origin}/forwarded.png"))
        .header("cdn-loop", format!("cloudflare, {other}"))
        .header("cdn-loop", "akamai;v=1.0")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        mark_for(&heads, "/forwarded.png").await,
        format!("cloudflare, {other}, akamai;v=1.0, {entry}")
    );
}

#[tokio::test]
async fn entries_past_the_forwarding_bound_are_not_carried_but_still_read() {
    let (origin, heads) = spawn_recording_origin().await;
    let base = spawn_app(true).await;
    let entry = own_entry(&base, &origin, &heads).await;
    let long = format!("cdn-{}", "x".repeat(3000));
    let resp = client()
        .get(format!("{base}/unsafe/20x0/{origin}/long.png"))
        .header("cdn-loop", &long)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // The fetch carries the server's entry alone.
    assert_eq!(mark_for(&heads, "/long.png").await, entry);
    // The server's entry after them is still found.
    let resp = client()
        .get(format!("{base}/unsafe/20x0/{origin}/long-marked.png"))
        .header("cdn-loop", format!("{long}, {entry}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::LOOP_DETECTED);
}
