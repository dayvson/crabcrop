//! Issue #475: one `AppState` per server, shared behind an
//! `Arc`. The router, every request, and every task a
//! request starts hold a pointer to it; none copies its
//! fields.

use super::*;

#[tokio::test]
async fn the_router_and_a_request_share_the_served_state() {
    // Copying the state took a reference to every `Arc` it
    // holds: the router took a copy for each route, and a
    // request one for its handler and one for each task it
    // started. Sharing it takes none, so the cache and the
    // metrics are held only by the one served state and by
    // this test, once the router is built and while a
    // request is parked on a source that accepts and never
    // answers.
    let source = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let src = format!("http://{}/src.png", source.local_addr().unwrap());
    let state = base_test_state(false, None);
    let cache = state.cache.clone();
    let metrics = state.metrics.clone();
    let base = spawn_state(state).await;
    // The router is built once the server task runs.
    let health = client().get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    let before = (Arc::strong_count(&cache), Arc::strong_count(&metrics));
    let remainder = format!("30x20/{src}");
    let request = tokio::spawn(async move { get_signed(&base, &remainder).await });
    let (_parked, _) = within(
        std::time::Duration::from_secs(10),
        "the request to reach its source",
        source.accept(),
    )
    .await
    .unwrap();
    let during = (Arc::strong_count(&cache), Arc::strong_count(&metrics));
    request.abort();
    assert_eq!(before, (2, 2), "the router copied the state");
    assert_eq!(during, (2, 2), "the request copied the state");
}

#[tokio::test]
async fn the_router_hands_each_handler_the_state_it_was_given() {
    // `app` keeps the `Arc` it is given, and the extractor
    // every handler takes clones that `Arc`, so a handler
    // reads the very state the server was built with.
    use axum::extract::FromRequestParts as _;
    let state = Arc::new(base_test_state(false, None));
    let router = app(state.clone());
    assert!(
        Arc::strong_count(&state) > 1,
        "the router holds no handle on the state"
    );
    let (mut parts, ()) = axum::http::Request::new(()).into_parts();
    let State(seen) = State::<Arc<AppState>>::from_request_parts(&mut parts, &state)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&seen, &state));
    drop((seen, router));
    assert_eq!(
        Arc::strong_count(&state),
        1,
        "the router kept a handle on the state"
    );
}

#[test]
fn an_engine_job_shares_the_states_providers() {
    // A job used to copy each provider handle out of the
    // state; it now takes the state's one `Arc` of them.
    let state = test_state();
    let work = MissWork {
        ops: Vec::new(),
        source_url: "example.com/i.png".to_string(),
        directives: crate::url::RequestDirectives::analyze(&[]),
        preferred: None,
        meta: false,
        debug: false,
        placeholders: false,
        hinted_width: None,
        dpr_hint: None,
        save_data: false,
        accept: None,
        groups: Vec::new(),
        cache_key: "key".to_string(),
        max_age: None,
        sealed: false,
        progress: MissProgress::default(),
        disk_handoff: None,
    };
    let job = build_job(&state, &work, Default::default());
    assert!(Arc::ptr_eq(&job.providers, &state.providers));
}
