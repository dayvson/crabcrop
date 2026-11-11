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
    // A whole-state copy no longer builds (`AppState` is not
    // `Clone`), but a handler or a task that clones the cache
    // or the metrics `Arc` itself also lands here: count it
    // in if that is meant.
    assert_eq!(
        during,
        (2, 2),
        "the request copied the state, or holds a clone of its cache or metrics"
    );
}

#[test]
fn the_router_holds_the_arc_it_is_given_and_releases_it() {
    // `app` keeps the `Arc` it is given, not a state of its
    // own, so the routes hold clones of the caller's `Arc`,
    // and dropping the router lets go of every one.
    let state = Arc::new(base_test_state(false, None));
    let router = app(state.clone());
    assert!(
        Arc::strong_count(&state) > 1,
        "the router holds no handle on the state"
    );
    drop(router);
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

/// How many `AppState` literals in `source` spell out every
/// field, with no `..` of their own to take the rest.
fn full_state_literals(source: &str) -> usize {
    let opening = "AppState {";
    let mut full = 0;
    let mut rest = source;
    while let Some(at) = rest.find(opening) {
        let before = rest[..at].trim_end();
        rest = &rest[at + opening.len()..];
        // A signature or a definition, not a literal.
        if ["->", "struct", "for", "impl"]
            .iter()
            .any(|word| before.ends_with(word))
        {
            continue;
        }
        // Read the literal to its closing brace, looking for
        // a `..` at its own level, outside any nested braces
        // or brackets.
        let mut depth = 0;
        let mut spread = false;
        for (i, c) in rest.char_indices() {
            match c {
                '{' | '(' | '[' => depth += 1,
                '}' if depth == 0 => break,
                '}' | ')' | ']' => depth -= 1,
                '.' if depth == 0 && rest[i..].starts_with("..") => spread = true,
                _ => {}
            }
        }
        if !spread {
            full += 1;
        }
    }
    full
}

/// Every `.rs` file under `dir`, at any depth.
fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read a source directory") {
        let path = entry.expect("read a directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn only_test_state_spells_out_every_field() {
    // A test names the fields it sets and takes the rest from
    // `test_state()`, so a field added to `AppState` is added
    // there, in `main.rs`, and in the one integration test
    // that cannot reach `test_state()`, not in every test.
    // Every source file in `src/` and `tests/` is read, so a
    // new file is counted too, except this one: its checks
    // below spell literals in strings.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    rust_files(&root.join("tests"), &mut files);
    let mut full: Vec<(String, usize)> = files
        .iter()
        .filter(|path| !path.ends_with("src/server/tests/shared_state.rs"))
        .map(|path| {
            let source = std::fs::read_to_string(path).expect("read a source file");
            let name = path
                .strip_prefix(root)
                .expect("a file under the package")
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            (name, full_state_literals(&source))
        })
        .filter(|(_, count)| *count > 0)
        .collect();
    full.sort();
    let expected = [
        "src/main.rs",
        "src/server/tests.rs",
        "tests/debug_redaction.rs",
    ]
    .map(|name| (name.to_string(), 1));
    assert_eq!(
        full, expected,
        "an AppState literal spells out every field outside main.rs, test_state() \
         and tests/debug_redaction.rs"
    );
    // The counter finds a full literal, also one with a `..`
    // nested in a field, and passes a spread and an `impl`.
    assert_eq!(full_state_literals("let s = AppState { a: 1, b: 2 };"), 1);
    assert_eq!(
        full_state_literals("AppState { a: S { x, ..S::default() }, b: f(0..2) }"),
        1
    );
    assert_eq!(
        full_state_literals("let s = AppState { a: 1, ..test_state() };"),
        0
    );
    assert_eq!(full_state_literals("impl AppState { fn f() {} }"), 0);
}
