//! Spellings of one source (issue #778). The loader fetches
//! one object for several spellings of a source: dot segments
//! and an upper-case scheme in an HTTP URL; a `localhost`
//! host, extra leading slashes, dot segments and percent
//! escapes in a `file://` path. Every spelling of one object
//! is keyed on the one form the loader fetches, so its
//! renders share an entry, and a purge or an overwrite that
//! names any spelling reaches it. Keys of sources already in
//! that form are what they were.

use super::*;

/// An answer read off the wire: its status, its head, and
/// its body.
struct Exact {
    status: u16,
    head: String,
    body: Vec<u8>,
}

impl Exact {
    /// The value of the header `name`, or empty when the
    /// answer has none.
    fn header(&self, name: &str) -> String {
        self.head
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.trim()
                    .eq_ignore_ascii_case(name)
                    .then(|| value.trim().to_string())
            })
            .unwrap_or_default()
    }
}

/// Send `method` `path` to `base` with the path bytes as
/// written: an HTTP client resolves the dot segments these
/// paths are signed with before it sends them.
async fn send_exact(base: &str, method: &str, path: &str) -> Exact {
    let addr = base.strip_prefix("http://").unwrap();
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    socket.write_all(request.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    socket.read_to_end(&mut raw).await.unwrap();
    let end = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no response head for {path}"));
    let head = String::from_utf8_lossy(&raw[..end]).into_owned();
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let exact = Exact {
        status,
        head,
        body: raw[end + 4..].to_vec(),
    };
    assert!(
        !exact.header("content-length").is_empty(),
        "{path}: a body this helper can read"
    );
    exact
}

/// The image path of `remainder`, signed as written.
fn signed_path(remainder: &str) -> String {
    format!("/{}/{remainder}", sign(remainder))
}

/// The purge path of `target`, signed as written.
fn purge_path(target: &str) -> String {
    format!(
        "/purge/{}/{target}",
        crate::url::purge_signature(SECRET, target)
    )
}

/// A PNG unlike [`test_png`]: white, and larger.
fn other_png() -> Vec<u8> {
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        200,
        120,
        image::Rgb([255, 255, 255]),
    ));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

/// Five spellings of `photo.png` the loader opens as one
/// file: the issue's list.
const FILE_SPELLINGS: [&str; 5] = [
    "file:///photo.png",
    "file://localhost/photo.png",
    "file:////photo.png",
    "file:///./photo.png",
    "file:///%70hoto.png",
];

#[tokio::test]
async fn an_overwrite_re_renders_every_spelling_of_the_replaced_file() {
    // Rendered under each spelling, then replaced by an
    // upload: every spelling answers the new image, with a
    // new ETag. The overwrite used to purge the entries of the
    // one spelling the upload hands back, and the other four
    // kept answering the old render.
    let dir = upload_dir("spellings-overwrite");
    let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, true).await;
    let name = "photo.png";
    let put = format!(
        "{base}/upload/{}/{name}",
        crate::url::upload_signature(SECRET, name)
    );
    let resp = client().put(&put).body(test_png()).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let mut before = Vec::new();
    for source in FILE_SPELLINGS {
        let answer = send_exact(&base, "GET", &signed_path(&format!("40x0/{source}"))).await;
        assert_eq!(answer.status, 200, "{source}");
        before.push(answer);
    }
    let resp = client().put(&put).body(other_png()).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    for (source, old) in FILE_SPELLINGS.iter().zip(&before) {
        let answer = send_exact(&base, "GET", &signed_path(&format!("40x0/{source}"))).await;
        assert_eq!(answer.status, 200, "{source}");
        assert_ne!(
            answer.header("etag"),
            old.header("etag"),
            "{source} answered the replaced render"
        );
        assert_ne!(answer.body, old.body, "{source}");
        let img = image::load_from_memory(&answer.body).unwrap();
        assert_eq!((img.width(), img.height()), (40, 24), "{source}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn spellings_of_one_source_share_one_entry() {
    // An HTTP source: a dot segment, a `..` segment and an
    // upper-case scheme are one request to the origin, so the
    // origin is asked once.
    let (src, hits) = spawn_counting_image_server().await;
    let base = spawn_app(false).await;
    let plain = format!("40x0/{src}");
    let resp = client()
        .get(format!("{base}{}", signed_path(&plain)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    for spelling in [
        src.replace("/src.png", "/./src.png"),
        src.replace("/src.png", "/x/../src.png"),
        src.replace("http://", "HTTP://"),
    ] {
        let answer = send_exact(&base, "GET", &signed_path(&format!("40x0/{spelling}"))).await;
        assert_eq!(answer.status, 200, "{spelling}");
        assert_eq!(answer.header("cache-status"), "crabcrop; hit", "{spelling}");
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "one fetch for every spelling"
    );
    // A file source: the five spellings open one file.
    let dir = upload_dir("spellings-share");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("photo.png"), test_png()).unwrap();
    let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, true).await;
    let first = send_exact(&base, "GET", &signed_path("40x0/file:///photo.png")).await;
    assert_eq!(first.status, 200);
    for source in &FILE_SPELLINGS[1..] {
        let answer = send_exact(&base, "GET", &signed_path(&format!("40x0/{source}"))).await;
        assert_eq!(answer.status, 200, "{source}");
        assert_eq!(answer.header("cache-status"), "crabcrop; hit", "{source}");
        assert_eq!(answer.header("etag"), first.header("etag"), "{source}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_purge_reaches_the_dot_segment_spelling_of_its_source() {
    // The issue's acceptance case: purging `w/http://h/a.jpg`
    // also removes `w/http://h/./a.jpg`. And the other way
    // round: a purge spelled with the dot segment removes the
    // plain spelling's entry.
    let (src, hits) = spawn_counting_image_server().await;
    let base = spawn_app(false).await;
    let plain = format!("40x0/{src}");
    let dotted = format!("40x0/{}", src.replace("/src.png", "/./src.png"));
    for remainder in [&plain, &dotted] {
        let answer = send_exact(&base, "GET", &signed_path(remainder)).await;
        assert_eq!(answer.status, 200, "{remainder}");
    }
    let purged = send_exact(&base, "POST", &purge_path(&plain)).await;
    assert_eq!(purged.status, 200);
    let fetched = hits.load(Ordering::SeqCst);
    let answer = send_exact(&base, "GET", &signed_path(&dotted)).await;
    assert_eq!(answer.status, 200);
    assert_eq!(
        answer.header("cache-status"),
        "crabcrop; fwd=uri-miss",
        "the dot-segment spelling outlived the purge"
    );
    assert_eq!(hits.load(Ordering::SeqCst), fetched + 1);
    // Now cached again; a purge spelled with the dot
    // segment reaches the plain spelling.
    let purged = send_exact(&base, "POST", &purge_path(&dotted)).await;
    assert_eq!(purged.status, 200);
    let answer = send_exact(&base, "GET", &signed_path(&plain)).await;
    assert_eq!(answer.status, 200);
    assert_eq!(
        answer.header("cache-status"),
        "crabcrop; fwd=uri-miss",
        "the plain spelling outlived the dotted purge"
    );
}

#[tokio::test]
async fn a_purge_reaches_every_spelling_of_its_file_source() {
    // A per-URL purge of one spelling of a file source removes
    // the render made under another.
    let dir = upload_dir("spellings-purge");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("photo.png"), test_png()).unwrap();
    let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, true).await;
    for (rendered, purged) in [
        ("file:///photo.png", "file://localhost/photo.png"),
        ("file:///%70hoto.png", "file:///photo.png"),
        ("file:////photo.png", "file:///./photo.png"),
    ] {
        let remainder = format!("40x0/{rendered}");
        let answer = send_exact(&base, "GET", &signed_path(&remainder)).await;
        assert_eq!(answer.status, 200, "{rendered}");
        let answer = send_exact(&base, "POST", &purge_path(&format!("40x0/{purged}"))).await;
        assert_eq!(answer.status, 200, "{purged}");
        let answer = send_exact(&base, "GET", &signed_path(&remainder)).await;
        assert_eq!(
            answer.header("cache-status"),
            "crabcrop; fwd=uri-miss",
            "the purge of {purged} missed the render of {rendered}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_source_purge_reaches_every_spelling_of_its_source() {
    // The purge an overwrite runs matches entries on their
    // source: whichever spelling the render and the purge
    // name, it removes the entry.
    let state = Arc::new(test_state());
    let ops = parse_operations_with_expiry("40x0/example.com/a.png")
        .unwrap()
        .0;
    for (rendered, purged) in [
        ("file://localhost/photo.png", "file:///photo.png"),
        ("file:///photo.png", "file:////./photo.png"),
        ("http://h/f/./a.jpg", "http://h/f/a.jpg"),
        ("http://h/f/a.jpg", "HTTP://H/f/x/../a.jpg"),
    ] {
        let key = cache_key_for("test", &ops, rendered, &plain_negotiation(&ops));
        state.cache.put(key.clone(), tier_test_entry());
        assert!(state.cache.get(&key).is_some());
        let removed = purge_source_everywhere(&state, purged).await;
        assert_eq!(removed, 1, "{purged} after {rendered}");
        assert!(state.cache.get(&key).is_none(), "{purged} after {rendered}");
    }
}

#[test]
fn keys_of_sources_in_their_fetched_form_do_not_change() {
    // Keys of sources the loader fetches as written, as main
    // keyed them: a change here would turn every entry the
    // disk and shared tiers hold into a miss.
    let padded = format!("http://origin{}t.png", "/".repeat(5_000));
    let digest = {
        let (ops, source, _, _) = parse_operations_with_expiry(&padded).unwrap();
        format!(
            "{}#cfgtest#source",
            crate::cache::bounded_key_base(crate::url::canonical_form(&ops, source), source)
        )
    };
    assert!(digest.starts_with("sha256 "), "{digest}");
    for (remainder, key) in [
        (
            "300x200/http://example.com/a.jpg",
            "300x200/http://example.com/a.jpg#cfgtest#source",
        ),
        (
            "300x200/filters:grayscale/https://cdn.example.com/images/photo.jpg",
            "300x200/filters:grayscale/https://cdn.example.com/images/photo.jpg#cfgtest#source",
        ),
        (
            "filters:format(webp):quality(80)/https://cdn.example.com:8443/a.jpg",
            "filters:format(webp):quality(80)/https://cdn.example.com:8443/a.jpg#cfgtest",
        ),
        (
            "fit-in:800x600/http://127.0.0.1:5341/f/src01.jpg",
            "fit-in:800x600/http://127.0.0.1:5341/f/src01.jpg#cfgtest#source",
        ),
        (
            "40x0/https://example.com/a.jpg?v=2&w=1",
            "40x0/https://example.com/a.jpg?v=2&w=1#cfgtest#source",
        ),
        (
            "40x0/https://example.com/with%20space/caf%C3%A9%2Fx.jpg",
            "40x0/https://example.com/with%20space/caf%C3%A9%2Fx.jpg#cfgtest#source",
        ),
        (
            "40x0/http://[::1]:8080/~user/a.png",
            "40x0/http://[::1]:8080/~user/a.png#cfgtest#source",
        ),
        (
            "40x0/https://xn--bcher-kva.example/a.jpg",
            "40x0/https://xn--bcher-kva.example/a.jpg#cfgtest#source",
        ),
        (
            "http://example.com/a.jpg",
            "http://example.com/a.jpg#cfgtest#source",
        ),
        (
            "50x30/file:///photo.png",
            "50x30/file:///photo.png#cfgtest#source",
        ),
        (
            "50x30/file:///heroes/cat-01_v2.jpg",
            "50x30/file:///heroes/cat-01_v2.jpg#cfgtest#source",
        ),
        (
            "50x30/file:///100%25.png",
            "50x30/file:///100%25.png#cfgtest#source",
        ),
        (
            "50x30/s3://my-bucket/photos/a.jpg",
            "50x30/s3://my-bucket/photos/a.jpg#cfgtest#source",
        ),
        (
            "50x30/gs://my-bucket/photos/a%20b.jpg",
            "50x30/gs://my-bucket/photos/a%20b.jpg#cfgtest#source",
        ),
        (
            "50x30/s3://my-bucket/a/./b.jpg",
            "50x30/s3://my-bucket/a/./b.jpg#cfgtest#source",
        ),
        // Text the loader fetches nothing for keeps its
        // spelling too.
        (
            "50x30/example.com/a.png",
            "50x30/example.com/a.png#cfgtest#source",
        ),
        (&padded, &digest),
    ] {
        let (ops, source, _, _) = parse_operations_with_expiry(remainder).unwrap();
        assert_eq!(
            cache_key_for("test", &ops, source, &plain_negotiation(&ops)),
            key,
            "{remainder}"
        );
    }
}
