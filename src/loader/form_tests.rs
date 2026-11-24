//! The fetched form of a source (issue #778) over awkward
//! spellings: one form names one object, a form is its own
//! form, and the forms of spellings the loader reads as text it
//! would rewrite (`\`, `?`, a tab, a line break, a trailing
//! space, a dot segment that escaped slashes leave, a drive
//! letter's `|`) read back as the same path.

use super::*;
use std::collections::{BTreeMap, BTreeSet};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// `s` with its control characters spelled out, for messages.
fn shown(s: &str) -> String {
    s.escape_debug().to_string()
}

/// The kind of a load error, without its message.
fn kind(e: &LoadError) -> String {
    format!("{e:?}").split('(').next().unwrap_or("").to_string()
}

/// File spellings: the issue's, the ones URL parsing rewrites,
/// and refused ones.
const FILE_SPELLINGS: &[&str] = &[
    "file:///A.jpg",
    "file:///%41.jpg",
    "file:///%2541.jpg",
    "file:///a#b.jpg",
    "file:///a%23b.jpg",
    "file:///a%2523b.jpg",
    "file:///100%.jpg",
    "file:///100%25.jpg",
    "file:///100%2525.jpg",
    "file:///a\\b.jpg",
    "file:///a%5Cb.jpg",
    "file:///a%5cb.jpg",
    "file:///a/b.jpg",
    "file:///a%2Fb.jpg",
    "file:///a%2fb.jpg",
    "file:///a%252Fb.jpg",
    "file:///a/./b.jpg",
    "file:///a%2f.%2fb.jpg",
    "file:///a//b.jpg",
    "file:///q?r.jpg",
    "file:///q%3Fr.jpg",
    "file:///q",
    "file:///C|/x.jpg",
    "file:///C%7C/x.jpg",
    "file:///C:/x.jpg",
    "file:///C%3A/x.jpg",
    "file:///caf%C3%A9.jpg",
    "file:///caf\u{e9}.jpg",
    "file:///cafe%CC%81.jpg",
    "file:///cafe\u{301}.jpg",
    "file:///%FF.jpg",
    "file:///%FE.jpg",
    "file:///%EF%BF%BD.jpg",
    "file:///\u{fffd}.jpg",
    "file:///sp ace.jpg",
    "file:///sp%20ace.jpg",
    "file:///tab%09.jpg",
    "file:///tab\t.jpg",
    "file:///tab.jpg",
    "file:///nl%0A.jpg",
    "file:///nl\n.jpg",
    "file:///nl.jpg",
    "file:///trail.jpg%20",
    "file:///trail.jpg ",
    "file:///trail.jpg",
    "file:///%20lead.jpg",
    "file:/// lead.jpg",
    "file:///lead.jpg",
    "file:///x.jpg",
    "file:///x.jpg?v=2",
    "file:///x.jpg%3Fv=2",
    "file:///x.jpg#f",
    "file:///x.jpg/",
    "file:///x.jpg%2F",
    "file:///./x.jpg",
    "file:///sub/../x.jpg",
    "file:///%2e/x.jpg",
    "file:///a/%2e%2e/x.jpg",
    "file:///../x.jpg",
    "file:///a%2f%2e%2e%2fx.jpg",
    "file:///..%2fx.jpg",
    "file:///.%2fx.jpg",
    "file://localhost/%2f%2E",
    "file:///x.jpg%00",
    "file://localhost/x.jpg",
    "file://LOCALHOST/x.jpg",
    "file://%6Cocalhost/x.jpg",
    "file://example.com/x.jpg",
    "file://example.com/C:/x.jpg",
    "file:////x.jpg",
    "file://\\x.jpg",
    "file:///e//x.jpg",
    "file:///e/x.jpg",
    "file:///plus+.jpg",
    "file:///plus%2B.jpg",
    "FILE:///x.jpg",
    "file:/x.jpg",
    " file:///x.jpg",
];

/// HTTP spellings, `{A}` standing for the origin's address and
/// `{P}` for its port.
const HTTP_SPELLINGS: &[&str] = &[
    "http://{A}/A.jpg",
    "http://{A}/%41.jpg",
    "http://{A}/a%2Fb.jpg",
    "http://{A}/a/b.jpg",
    "http://{A}/a%2fb.jpg",
    "http://{A}/%25.jpg",
    "http://{A}/%.jpg",
    "http://{A}/x.jpg",
    "http://{A}/x.jpg?",
    "http://{A}/x.jpg#f",
    "http://{A}/x.jpg#",
    "http://{A}/x.jpg?#f",
    "http://{A}/x.jpg?a=1&b=2",
    "http://{A}/x.jpg?b=2&a=1",
    "http://{A}\\a\\b.jpg",
    "http://{A}/a/./b.jpg",
    "http://{A}/a/c/../b.jpg",
    "http://{A}/a/%2e/b.jpg",
    "http://{A}/a/c/%2E%2e/b.jpg",
    "http://{A}/a\tb.jpg",
    "http://{A}/a\nb.jpg",
    "http://{A}/ab.jpg",
    "http://{A}/a b.jpg",
    "http://{A}/a%20b.jpg",
    "http://{A}/caf%C3%A9.jpg",
    "http://{A}/caf\u{e9}.jpg",
    "http://{A}/a|b.jpg",
    "http://{A}/a%7Cb.jpg",
    "http://{A}/a^b.jpg",
    "http://{A}/a%5Eb.jpg",
    "http://{A}/a{b}.jpg",
    "http://{A}/a%7Bb%7D.jpg",
    "http://{A}/x.jpg?q=a b",
    "http://{A}/x.jpg?q=a%20b",
    "http://{A}/x.jpg?q='a'",
    "http://{A}/x.jpg?q=%27a%27",
    "http://{A}/x.jpg?q=a+b",
    "http://{A}/~u.jpg",
    "http://{A}/%7Eu.jpg",
    "HTTP://{A}/x.jpg",
    "http://@{A}/x.jpg",
    "http://:@{A}/x.jpg",
    "http://u@{A}/x.jpg",
    "http:{A}/x.jpg",
    "http:/{A}/x.jpg",
    "http:///{A}/x.jpg",
    " http://{A}/x.jpg",
    "http://127.1:{P}/x.jpg",
    "http://0x7f.0.0.1:{P}/x.jpg",
    "http://127.0.0.1:0{P}/x.jpg",
    "http://[fe80::1%25eth0]:{P}/x.jpg",
];

/// Assert that `form`, the fetched form of `spelling`, is its
/// own fetched form, and that, for a file source the loader
/// reads, it reads back as the same path.
fn assert_fixed(spelling: &str, form: &str) {
    let again = fetched_form(form);
    assert_eq!(
        again,
        form,
        "{} => {} => {}",
        shown(spelling),
        shown(form),
        shown(&again)
    );
    if spelling.starts_with("file://")
        && let Ok(relative) = file_source_relative(spelling)
    {
        assert_eq!(
            file_source_relative(form).as_deref(),
            Ok(relative.as_str()),
            "{} => {}",
            shown(spelling),
            shown(form)
        );
    }
}

#[test]
fn a_fetched_form_is_its_own_fetched_form() {
    // A form printed by `/explain`, or written into a URL, names
    // the object of the spelling it came from: requested or
    // purged, it is keyed the same.
    for &spelling in FILE_SPELLINGS {
        assert_fixed(spelling, &fetched_form(spelling));
    }
    for &template in HTTP_SPELLINGS {
        let spelling = template
            .replace("{A}", "127.0.0.1:5341")
            .replace("{P}", "5341");
        assert_fixed(&spelling, &fetched_form(&spelling));
    }
    for spelling in [
        "s3://bucket/key with space.jpg",
        "gs://bucket/caf\u{e9}.jpg",
        "example.com/a.png",
        "root:photos/a.jpg",
    ] {
        assert_fixed(spelling, &fetched_form(spelling));
    }
}

#[test]
fn the_text_url_parsing_rewrites_is_escaped_in_a_file_form() {
    // Each of these decodes to a path URL parsing would read
    // differently if the form held it raw. Escaped, the form is
    // the spelling the loader read, as main keyed it.
    for spelling in [
        "file:///a%5Cb.jpg",
        "file:///q%3Fr.jpg",
        "file:///tab%09.jpg",
        "file:///nl%0A.jpg",
        "file:///trail.jpg%20",
        "file:///C%7C/x.jpg",
        "file:///a%2523b.jpg",
    ] {
        assert!(
            matches!(fetched_form(spelling), std::borrow::Cow::Borrowed(_)),
            "{spelling}: {}",
            fetched_form(spelling)
        );
    }
    for (spelling, form) in [
        ("file:///a%5cb.jpg", "file:///a%5Cb.jpg"),
        ("file:///a%2f.%2fb.jpg", "file:///a%2F.%2Fb.jpg"),
        ("file:///.%2fx.jpg", "file:///%2F.%2Fx.jpg"),
        ("file://localhost/%2f%2E", "file:///%2F."),
        // A space inside a name and non-ASCII text are keyed
        // decoded, as before: URL parsing escapes them itself.
        ("file:///sp%20ace.jpg", "file:///sp ace.jpg"),
        ("file:///caf%C3%A9.jpg", "file:///caf\u{e9}.jpg"),
    ] {
        assert_eq!(fetched_form(spelling), form, "{spelling}");
    }
}

#[tokio::test]
async fn the_spellings_that_share_a_form_open_one_file() {
    // A tree of awkward names: the spellings that share a
    // fetched form all load the same file, or are all refused,
    // and no form of a file the loader opens holds a `#`.
    let base = crate::test_support::scratch_dir("loader-form-files");
    let root = base.join("root");
    std::fs::create_dir_all(&root).unwrap();
    for name in [
        "A.jpg",
        "%41.jpg",
        "a#b.jpg",
        "a%23b.jpg",
        "100%.jpg",
        "100%25.jpg",
        "a\\b.jpg",
        "a/b.jpg",
        "a%2Fb.jpg",
        "q",
        "q?r.jpg",
        "C|/x.jpg",
        "C:/x.jpg",
        "caf\u{e9}.jpg",
        "cafe\u{301}.jpg",
        "\u{fffd}.jpg",
        "sp ace.jpg",
        "tab\t.jpg",
        "tab.jpg",
        "nl\n.jpg",
        "nl.jpg",
        "trail.jpg ",
        "trail.jpg",
        " lead.jpg",
        "lead.jpg",
        "x.jpg",
        "x.jpg?v=2",
        "e/x.jpg",
        "plus+.jpg",
    ] {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, name).unwrap();
    }
    let loader = Loader::new(1 << 20, Duration::from_secs(5))
        .unwrap()
        .with_sources(SourceConfig {
            file_root: Some(std::fs::canonicalize(&root).unwrap()),
            ..Default::default()
        });
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for &spelling in FILE_SPELLINGS {
        let form = fetched_form(spelling).into_owned();
        let outcome = match loader.load(spelling).await {
            Ok(bytes) => format!("Ok({})", String::from_utf8_lossy(&bytes).escape_debug()),
            Err(e) => format!("Err({})", kind(&e)),
        };
        if outcome.starts_with("Ok") {
            assert!(!form.contains('#'), "{}: {}", shown(spelling), shown(&form));
            // The form loads the file the spelling did.
            let again = match loader.load(&form).await {
                Ok(bytes) => format!("Ok({})", String::from_utf8_lossy(&bytes).escape_debug()),
                Err(e) => format!("Err({})", kind(&e)),
            };
            assert_eq!(again, outcome, "{} => {}", shown(spelling), shown(&form));
        }
        groups
            .entry(form)
            .or_default()
            .push((spelling.to_string(), outcome));
    }
    let shared: Vec<String> = groups
        .iter()
        .filter(|(_, members)| {
            members
                .iter()
                .map(|(_, outcome)| outcome.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                > 1
        })
        .map(|(form, members)| format!("{}: {members:?}", shown(form)))
        .collect();
    assert!(
        shared.is_empty(),
        "one form, several objects:\n{}",
        shared.join("\n")
    );
    let _ = std::fs::remove_dir_all(&base);
}

/// A one-shot origin that records the request head it gets.
async fn record_one_request() -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = captured.clone();
    tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            let n = socket.read(&mut chunk).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        *sink.lock().unwrap() = String::from_utf8_lossy(&buf).into_owned();
        let mut out = b"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\nConnection: close\r\n\r\n".to_vec();
        out.extend_from_slice(b"\xff\xd8\xff\xd9");
        let _ = socket.write_all(&out).await;
    });
    (addr.to_string(), captured)
}

#[tokio::test]
async fn the_spellings_that_share_a_form_make_one_request() {
    // Against a recording origin: the spellings that share a
    // fetched form send the same request line and Host header,
    // and no form of a request the loader makes holds a `#`.
    let loader = Loader::with_defaults().unwrap();
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for &template in HTTP_SPELLINGS {
        let (addr, captured) = record_one_request().await;
        let port = addr.rsplit(':').next().unwrap().to_string();
        let url = template.replace("{A}", &addr).replace("{P}", &port);
        let form = fetched_form(&url).replace(&addr, "{A}");
        let outcome = match loader.load(&url).await {
            Ok(_) => {
                let head = captured.lock().unwrap().clone();
                let target = head.split_whitespace().nth(1).unwrap_or("").to_string();
                let host = head
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.trim()
                            .eq_ignore_ascii_case("host")
                            .then(|| value.trim().to_string())
                    })
                    .unwrap_or_default()
                    .replace(&addr, "{A}");
                format!("GET {target} host={host}")
            }
            Err(e) => format!("Err({})", kind(&e)),
        };
        if outcome.starts_with("GET") {
            assert!(!form.contains('#'), "{url}: {form}");
        }
        groups
            .entry(form)
            .or_default()
            .push((template.to_string(), outcome));
    }
    let shared: Vec<String> = groups
        .iter()
        .filter(|(_, members)| {
            members
                .iter()
                .map(|(_, outcome)| outcome.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                > 1
        })
        .map(|(form, members)| format!("{}: {members:?}", shown(form)))
        .collect();
    assert!(
        shared.is_empty(),
        "one form, several requests:\n{}",
        shared.join("\n")
    );
}

/// A small xorshift generator, so a run is the same every time.
struct Generator(u64);

impl Generator {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[(self.next() % from.len() as u64) as usize]
    }
}

/// What the loader does with `s`, from its text alone: the
/// root-relative path it opens, the URL it requests, or a
/// refusal (`None`).
fn object(s: &str) -> Option<String> {
    if s.starts_with("s3://") || s.starts_with("gs://") {
        return Some(format!("bucket:{s}"));
    }
    if s.starts_with("file://") {
        return file_source_relative(s)
            .ok()
            .map(|relative| format!("file:{relative}"));
    }
    let mut parsed = Url::parse(s).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    parsed.set_fragment(None);
    Some(format!("http:{parsed}"))
}

/// Generate `rounds` spellings from `prefixes` and `tokens` and
/// check every fetched form: one object per form, no `#` in the
/// form of a source the loader fetches, and each form its own
/// form, naming the same object.
fn check_generated(prefixes: &[&str], tokens: &[&str], rounds: usize, seed: u64) {
    let mut generator = Generator(seed);
    let mut by_form: std::collections::HashMap<String, (String, Option<String>)> =
        std::collections::HashMap::new();
    let mut found = Vec::new();
    for _ in 0..rounds {
        let mut s = generator.pick(prefixes).to_string();
        for _ in 0..(generator.next() % 9) {
            s.push_str(generator.pick(tokens));
        }
        let form = fetched_form(&s).into_owned();
        let target = object(&s);
        let bucket = s.starts_with("s3://") || s.starts_with("gs://");
        if target.is_some() && !bucket && form.contains('#') {
            found.push(format!(
                "a form with '#': {} => {}",
                shown(&s),
                shown(&form)
            ));
        }
        let again = fetched_form(&form);
        if again != form {
            found.push(format!(
                "not its own form: {} => {} => {}",
                shown(&s),
                shown(&form),
                shown(&again)
            ));
        } else if target.is_some() && object(&form) != target {
            found.push(format!(
                "the form names another object: {} => {}",
                shown(&s),
                shown(&form)
            ));
        }
        match by_form.get(&form) {
            Some((other, other_target)) if *other_target != target => {
                found.push(format!(
                    "{} [{other_target:?}] and {} [{target:?}] share {}",
                    shown(other),
                    shown(&s),
                    shown(&form)
                ));
            }
            Some(_) => {}
            None => {
                by_form.insert(form, (s, target));
            }
        }
    }
    assert!(
        found.is_empty(),
        "{}",
        found[..found.len().min(10)].join("\n")
    );
}

#[test]
fn generated_file_spellings_keep_one_object_per_form() {
    let prefixes = [
        "file:///",
        "file://",
        "file:////",
        "file://localhost/",
        "file://localhost",
        "file://LOCALHOST/",
        "file://%6Cocalhost/",
        "file://h/",
        "file:",
        "file:/",
        "FILE:///",
        " file:///",
    ];
    let tokens = [
        "a",
        "b",
        ".",
        "..",
        "/",
        "//",
        "%",
        "%2e",
        "%2E",
        "%2f",
        "%2F",
        "%23",
        "%25",
        "%2525",
        "%3F",
        "%3f",
        "%5C",
        "\\",
        "?",
        "#",
        " ",
        "%20",
        "%00",
        "%41",
        "A",
        "%7C",
        "|",
        "C",
        ":",
        "%3A",
        "\t",
        "\n",
        "%FF",
        "%C3%A9",
        "\u{e9}",
        "%0A",
        "x.jpg",
        "%2e%2e",
        ".%2e",
        "localhost",
        "%09",
        "%0D",
        "%01",
    ];
    check_generated(&prefixes, &tokens, 400_000, 0x9E37_79B9_7F4A_7C15);
}

#[test]
fn generated_http_spellings_keep_one_object_per_form() {
    let prefixes = [
        "http://h",
        "HTTP://H",
        "http://h:80",
        "https://h:443",
        "http://h:8080",
        "http:/h",
        "http:h",
        "http:///h",
        "http://@h",
        "http://u@h",
        "http://u:p@h",
        " http://h",
        "http://h.",
        "http://[::1]",
        "http://[fe80::1%25eth0]",
        "http://xn--bcher-kva.example",
        "http://b\u{fc}cher.example",
        "http://127.1",
        "http://0x7f.1",
        "http://h%2Ex",
        "s3://bucket",
        "gs://bucket",
        "ftp://h",
    ];
    let tokens = [
        "/", "a", "A", ".", "..", "%2e", "%2E", "%2f", "%2F", "%41", "?", "#", "&", "=", "+",
        "%20", " ", "\\", "\t", "\n", "%", "%25", "%23", "@", ":", "80", "[", "]", "\u{e9}", "{",
        "}", "|", "^", "`", "\"", "<", ">", "'", "%7E", "~", "%00",
    ];
    check_generated(&prefixes, &tokens, 400_000, 0xD1B5_4A32_D192_ED03);
}
