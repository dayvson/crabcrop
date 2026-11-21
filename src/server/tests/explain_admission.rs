//! `/explain` and the image route agree on admission (issue
//! #741). The explain route admits through the image path's
//! own [`authenticate`] and [`resolve_request`], so a URL is
//! admitted by both routes or refused by both, with one answer.
//! The test sends every case of `spec/explain-verdicts.json`,
//! and the expiry, key, sealing, preset, default-operation and
//! source-root cases beside it, to both routes of one server
//! whose source exists, and compares the answers.
//!
//! [`authenticate`]: crate::server::admission::authenticate

use super::*;

/// The test server's rotation window (issue #143): the test
/// secret is the current key, under [`CURRENT_ID`], and this is
/// the previous one, under [`PREVIOUS_ID`].
const PREVIOUS_SECRET: &[u8] = b"the-previous-test-secret";
const CURRENT_ID: &str = "k-new";
const PREVIOUS_ID: &str = "k-old";

/// The test server's default operations (issue #145): one
/// operation, which a URL's own quality drops.
const DEFAULT_OPS: &str = "filters:quality(80)";

/// The test server: a rotation window over the test secret,
/// presets, the default operations and a source root on the
/// origin.
fn admission_state(origin: &str) -> AppState {
    let preset = |text: &str| text.parse::<crate::url::Preset>().unwrap();
    let presets: crate::url::Presets = [
        ("thumb", preset("300x200/filters:quality(80)")),
        ("gray", preset("filters:grayscale")),
        // Two references run past the operation cap.
        (
            "heavy",
            preset(&format!("filters:{}", ["invert"; 60].join(":"))),
        ),
    ]
    .into_iter()
    .map(|(name, preset)| (name.to_string(), preset))
    .collect();
    let default_ops = crate::url::parse_default_ops(DEFAULT_OPS, &presets).unwrap();
    AppState {
        keyring: crate::url::Keyring {
            current_id: CURRENT_ID.to_string(),
            current: SECRET.to_vec(),
            previous: Some((PREVIOUS_ID.to_string(), PREVIOUS_SECRET.to_vec())),
        },
        presets,
        default_ops,
        source_roots: [("origin".to_string(), origin.to_string())]
            .into_iter()
            .collect(),
        ..test_state()
    }
}

/// The paths of the requests an origin was sent.
type Fetches = Arc<std::sync::Mutex<Vec<String>>>;

/// An origin that serves the test PNG at every path and notes
/// the path of every request it is sent, so a test can tell
/// which case's source was fetched.
async fn spawn_recording_origin() -> (String, Fetches) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let fetches: Fetches = Arc::default();
    let noted = fetches.clone();
    let png = Arc::new(test_png());
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let (noted, png) = (noted.clone(), png.clone());
            tokio::spawn(async move {
                let mut head = Vec::new();
                let mut buf = [0u8; 1024];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < 64 * 1024 {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => head.extend_from_slice(&buf[..n]),
                    }
                }
                let line = String::from_utf8_lossy(&head);
                let path = line.split(' ').nth(1).unwrap_or("");
                noted.lock().unwrap().push(path.to_string());
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    png.len()
                );
                let _ = socket.write_all(reply.as_bytes()).await;
                let _ = socket.write_all(&png).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (format!("http://{addr}"), fetches)
}

/// One URL, sent to both routes.
struct Case {
    /// What the case is, for the failure messages.
    name: String,
    /// The signature segment (the token, for a sealed case)
    /// and the remainder: the path after the base URL.
    path: String,
    /// The last path segment of the case's source, when its
    /// source is the origin's. No other case fetches it, and
    /// the image route has no render of it stored. (Not a
    /// query string: the image route reads only the path.)
    marker: Option<String>,
    /// Whether `path` is a sealed spelling.
    sealed: bool,
    /// What the image route answers, for the cases added
    /// here: `"admitted"`, `"refused"`, or the code it
    /// refuses with. The corpus pins its own verdicts
    /// elsewhere.
    expect: Option<&'static str>,
}

/// One route's answer to a case.
#[derive(Debug)]
struct Answer {
    status: StatusCode,
    /// The code of an error in the JSON envelope.
    code: Option<String>,
    /// The envelope's message, or the body.
    message: String,
    /// Whether the origin was sent the case's source while
    /// the route answered.
    fetched: bool,
}

/// Every code the URL grammar refuses with
/// ([`crate::url::ParseError::code`]).
fn grammar_codes() -> std::collections::HashSet<&'static str> {
    use crate::url::ParseError as E;
    let text = String::new;
    [
        E::EmptyPath,
        E::MissingSignature,
        E::MalformedSignature,
        E::InvalidSize(text()),
        E::InvalidCrop(text()),
        E::InvalidCropRect(text()),
        E::InvalidFocalRect(text()),
        E::InvalidFocalPoint(text()),
        E::InvalidPad(text()),
        E::InvalidExpiry(text()),
        E::InvalidMaxAge(text()),
        E::InvalidDebug(text()),
        E::InvalidFrame(text()),
        E::InvalidPage(text()),
        E::UnknownPreset(text()),
        E::UnknownRoot(text()),
        E::InvalidRoot(text()),
        E::InvalidPreset(text()),
        E::PresetsOnly(text()),
        E::InvalidFilters(text()),
        E::InvalidCurve(text()),
        E::InvalidConvolution(text()),
        E::InvalidNoise(text()),
        E::MissingSourceUrl,
        E::TooManyOperations(0),
        E::ExpansionTooLong(0),
        E::TooManyWatermarks(0),
    ]
    .iter()
    .map(E::code)
    .collect()
}

impl Answer {
    /// Whether the image path's admission refused the
    /// request, as its code reads.
    ///
    /// [`authenticate`] refuses a signature segment that does
    /// not parse (`400` with the grammar's code), a remainder
    /// that begins with the reserved segment (`400
    /// reserved_segment`) and a signature that does not verify
    /// (`403 bad_signature`). A sealed segment it cannot open
    /// is the uniform refusal, `403` with the plain-text body
    /// `bad signature` whatever the `Accept`.
    ///
    /// [`resolve_request`] refuses a remainder that begins with
    /// a purge or upload segment (`400 reserved_segment`), a
    /// preset, grammar, operation-cap or source-root failure
    /// (`400` with the grammar's code), and an expired URL
    /// (`403 signature_expired`). For an opened token every
    /// one of these is the uniform refusal but the expiry,
    /// which is the plain-text `signature expired`.
    ///
    /// Both stages run before anything touches the source, so
    /// an answer that fetched it was admitted: the engine's own
    /// `400 unsupported_operation` comes after the fetch.
    ///
    /// [`authenticate`]: crate::server::admission::authenticate
    fn refused(&self) -> bool {
        let code = self.code.as_deref();
        let refusal = match (self.status, code) {
            (StatusCode::BAD_REQUEST, Some(code)) => {
                code == "reserved_segment" || grammar_codes().contains(code)
            }
            (StatusCode::FORBIDDEN, Some(code)) => {
                matches!(code, "bad_signature" | "signature_expired")
            }
            (StatusCode::FORBIDDEN, None) => {
                matches!(self.message.as_str(), "bad signature" | "signature expired")
            }
            _ => false,
        };
        refusal && !self.fetched
    }
}

/// `GET` `path` from `base` as written: an HTTP client
/// resolves dot segments (`..`, and `%2e%2e` with them) before
/// it sends a path, and these paths are signed as written.
/// Returns the status and the body.
async fn get_exact(base: &str, path: &str, accept: Option<&str>) -> (StatusCode, String) {
    let addr = base.strip_prefix("http://").unwrap();
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    let accept = accept.map_or(String::new(), |accept| format!("Accept: {accept}\r\n"));
    let request =
        format!("GET /{path} HTTP/1.1\r\nHost: {addr}\r\n{accept}Connection: close\r\n\r\n");
    socket.write_all(request.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    socket.read_to_end(&mut raw).await.unwrap();
    let raw = String::from_utf8_lossy(&raw);
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .and_then(|code| StatusCode::from_u16(code).ok())
        .unwrap_or_else(|| panic!("no status in {head}"));
    (status, body.to_string())
}

/// Send `path` to `base` and read the answer, noting whether
/// the origin was sent the source marked `marker` meanwhile.
async fn answer(
    base: &str,
    path: &str,
    accept: Option<&str>,
    fetches: &Fetches,
    marker: Option<&str>,
) -> Answer {
    let count = || {
        marker.map_or(0, |marker| {
            fetches
                .lock()
                .unwrap()
                .iter()
                .filter(|path| path.rsplit('/').next() == Some(marker))
                .count()
        })
    };
    let before = count();
    let (status, body) = get_exact(base, path, accept).await;
    let envelope = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .filter(|json| json["error"].is_object());
    let (code, message) = match envelope {
        Some(json) => (
            json["error"]["code"].as_str().map(str::to_string),
            json["error"]["message"].as_str().unwrap_or("").to_string(),
        ),
        None => (None, body.chars().take(200).collect()),
    };
    Answer {
        status,
        code,
        message,
        fetched: count() > before,
    }
}

/// `remainder` as an HTTP client sends it in a path: the bytes
/// a request target cannot carry (controls, the space,
/// ``"#<>?`{}`` and every byte past ASCII) percent-encoded.
/// The server reads the path as it was sent, so this is the
/// text a URL signs.
fn as_sent(remainder: &str) -> String {
    let mut out = String::with_capacity(remainder.len());
    for &byte in remainder.as_bytes() {
        if byte <= b' ' || byte >= 0x7f || b"\"#<>?`{}".contains(&byte) {
            out.push_str(&format!("%{byte:02X}"));
        } else {
            out.push(char::from(byte));
        }
    }
    out
}

/// The corpus cases, signed under the test secret, each
/// `https://example.com/` source moved onto the origin under
/// a marker of its own.
fn corpus_cases(origin: &str) -> Vec<Case> {
    let doc = load_spec("explain-verdicts.json");
    let cases = doc["cases"].as_array().unwrap();
    cases
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let remainder = case["remainder"].as_str().unwrap();
            let source = case["source"].as_str().unwrap();
            let ops = remainder
                .strip_suffix(source)
                .unwrap_or_else(|| panic!("{remainder} does not end in {source}"));
            let (remainder, marker) = match source.strip_prefix("https://example.com/") {
                Some(path) => {
                    let marker = format!("corpus{i}");
                    (format!("{ops}{origin}/{path}/{marker}"), Some(marker))
                }
                None => (remainder.to_string(), None),
            };
            let remainder = as_sent(&remainder);
            Case {
                name: format!("corpus case {i}"),
                path: format!("{}/{remainder}", sign(&remainder)),
                marker,
                sealed: false,
                expect: None,
            }
        })
        .collect()
}

/// The cases this file adds to the corpus's.
struct Added<'a> {
    origin: &'a str,
    cases: Vec<Case>,
}

impl Added<'_> {
    /// A source on the origin, under a marker no other case
    /// uses, and the marker.
    fn source(&self) -> (String, String) {
        let marker = format!("added{}", self.cases.len());
        (format!("{}/{marker}", self.origin), marker)
    }

    /// A plain case: `remainder` builds the remainder around
    /// a fresh source, and `signature` its signature segment.
    fn plain(
        &mut self,
        name: &str,
        expect: &'static str,
        remainder: impl FnOnce(&str) -> String,
        signature: impl FnOnce(&str) -> String,
    ) {
        let (source, marker) = self.source();
        let remainder = remainder(&source);
        self.cases.push(Case {
            name: name.to_string(),
            path: format!("{}/{remainder}", signature(&remainder)),
            marker: Some(marker),
            sealed: false,
            expect: Some(expect),
        });
    }

    /// A sealed case: `token` seals the remainder built
    /// around a fresh source.
    fn sealed(
        &mut self,
        name: &str,
        expect: &'static str,
        remainder: impl FnOnce(&str) -> String,
        token: impl FnOnce(&str) -> String,
    ) {
        let (source, marker) = self.source();
        let remainder = remainder(&source);
        self.cases.push(Case {
            name: name.to_string(),
            path: token(&remainder),
            marker: Some(marker),
            sealed: true,
            expect: Some(expect),
        });
    }
}

/// `remainder` signed under `secret`, with `id` stamped on the
/// signature segment.
fn keyed(id: &str, secret: &[u8], remainder: &str) -> String {
    format!("{id}:{}", sign_with(secret, remainder))
}

/// `remainder` sealed under `secret`.
fn seal(secret: &[u8], remainder: &str) -> String {
    crabcrop_url::seal(secret, remainder).unwrap()
}

/// `remainder` sealed under `secret`, with `id` stamped on the
/// token.
fn seal_keyed(id: &str, secret: &[u8], remainder: &str) -> String {
    crabcrop_url::seal_with_key_id(secret, id, remainder).unwrap()
}

/// A resize of `source`, the remainder most cases vary.
fn sized(source: &str) -> String {
    format!("300x200/{source}")
}

/// One hundred operations: at the cap alone, past it with the
/// server's default operation.
fn at_cap() -> String {
    format!("filters:{}", ["invert"; 100].join(":"))
}

/// Expired, keyed, sealed, preset, default-operation and
/// source-root cases, and the signature failures, each with
/// the answer it must get.
fn added_cases(origin: &str) -> Vec<Case> {
    let mut added = Added {
        origin,
        cases: Vec::new(),
    };
    expiry_cases(&mut added);
    key_cases(&mut added);
    signature_cases(&mut added);
    preset_cases(&mut added);
    default_and_root_cases(&mut added);
    sealed_cases(&mut added);
    added.cases
}

/// Expiry: checked after the signature, against the clock,
/// strictly before the `exp` second.
fn expiry_cases(added: &mut Added) {
    let now = now_unix();
    added.plain("a plain URL", "admitted", sized, sign);
    added.plain(
        "expired an hour ago",
        "signature_expired",
        |src| format!("exp:{}/300x200/{src}", now - 3600),
        sign,
    );
    added.plain(
        "expired at the first second",
        "signature_expired",
        |src| format!("exp:1/300x200/{src}"),
        sign,
    );
    added.plain(
        "expired, with a cache lifetime",
        "signature_expired",
        |src| format!("maxage:60/exp:{}/300x200/{src}", now - 1),
        sign,
    );
    added.plain(
        "expires in an hour",
        "admitted",
        |src| format!("exp:{}/300x200/{src}", now + 3600),
        sign,
    );
}

/// Keys: a keyed signature verifies under the named key only,
/// an unkeyed one under either.
fn key_cases(added: &mut Added) {
    added.plain("the current key ID", "admitted", sized, |r| {
        keyed(CURRENT_ID, SECRET, r)
    });
    added.plain("the previous key ID", "admitted", sized, |r| {
        keyed(PREVIOUS_ID, PREVIOUS_SECRET, r)
    });
    added.plain("unkeyed, under the previous key", "admitted", sized, |r| {
        sign_with(PREVIOUS_SECRET, r)
    });
    added.plain(
        "the previous key under the current ID",
        "bad_signature",
        sized,
        |r| keyed(CURRENT_ID, PREVIOUS_SECRET, r),
    );
    added.plain(
        "the current key under the previous ID",
        "bad_signature",
        sized,
        |r| keyed(PREVIOUS_ID, SECRET, r),
    );
    added.plain("an unknown key ID", "bad_signature", sized, |r| {
        keyed("k-gone", SECRET, r)
    });
    added.plain("a malformed key ID", "malformed_signature", sized, |r| {
        keyed("k.new", SECRET, r)
    });
    added.plain(
        "expired, under the previous key ID",
        "signature_expired",
        |src| format!("exp:1/300x200/{src}"),
        |r| keyed(PREVIOUS_ID, PREVIOUS_SECRET, r),
    );
    added.plain(
        "a grammar failure, under the current key ID",
        "invalid_size",
        |src| format!("fit-in:-300x200/{src}"),
        |r| keyed(CURRENT_ID, SECRET, r),
    );
}

/// Signatures that do not verify or do not parse, and the
/// reserved segments, refused whatever the signature.
fn signature_cases(added: &mut Added) {
    added.plain(
        "another remainder's signature",
        "bad_signature",
        sized,
        |_| sign("300x201/https://example.com/a.jpg"),
    );
    added.plain(
        "an uppercase signature",
        "malformed_signature",
        sized,
        |r| sign(r).to_uppercase(),
    );
    added.plain(
        "unsafe, on a server that refuses it",
        "bad_signature",
        sized,
        |_| "unsafe".to_string(),
    );
    added.plain(
        "a purge segment",
        "reserved_segment",
        |src| format!("purge/300x200/{src}"),
        sign,
    );
    added.plain(
        "the reserved segment",
        "reserved_segment",
        |src| format!("crabcrop/{src}"),
        sign,
    );
}

/// Presets, expanded after the signature.
fn preset_cases(added: &mut Added) {
    added.plain(
        "a preset",
        "admitted",
        |src| format!("preset:thumb/{src}"),
        sign,
    );
    added.plain(
        "two presets",
        "admitted",
        |src| format!("preset:gray/preset:thumb/{src}"),
        sign,
    );
    added.plain(
        "a preset and an operation",
        "admitted",
        |src| format!("preset:thumb/filters:blur(2)/{src}"),
        sign,
    );
    added.plain(
        "an unknown preset",
        "unknown_preset",
        |src| format!("preset:missing/{src}"),
        sign,
    );
    added.plain(
        "presets past the operation cap",
        "unsupported_operation",
        |src| format!("preset:heavy/preset:heavy/{src}"),
        sign,
    );
    added.plain(
        "a preset reference in the source path",
        "admitted",
        |src| {
            let (base, marker) = src.rsplit_once('/').unwrap();
            format!("300x200/{base}/preset:thumb/{marker}")
        },
        sign,
    );
}

/// Default operations, merged ahead of the URL's own and
/// counted with them, and source roots, expanded after the
/// signature.
fn default_and_root_cases(added: &mut Added) {
    let origin = added.origin;
    added.plain(
        "the defaults take a full list past the cap",
        "unsupported_operation",
        |src| format!("{}/{src}", at_cap()),
        sign,
    );
    added.plain(
        "the URL's own quality drops the default",
        "admitted",
        |src| format!("filters:quality(70):{}/{src}", ["invert"; 99].join(":")),
        sign,
    );
    added.plain("the defaults alone", "admitted", str::to_string, sign);
    added.plain(
        "a root source",
        "admitted",
        |src| sized(&src.replace(origin, "root:origin")),
        sign,
    );
    added.plain(
        "an unknown root",
        "unknown_root",
        |_| "300x200/root:elsewhere/src.png".to_string(),
        sign,
    );
    added.plain(
        "a root path that climbs out",
        "invalid_root",
        |_| "300x200/root:origin/../src.png".to_string(),
        sign,
    );
}

/// A sealed case: its name, what the image route answers,
/// and its remainder, from the origin and a fresh source.
type SealedPayload = (&'static str, &'static str, fn(&str, &str) -> String);

/// Sealed spellings: the image route opens them and admits
/// what their remainder resolves to; `/explain` refuses every
/// one (issue #255).
fn sealed_cases(added: &mut Added) {
    let origin = added.origin;
    let sealed = |r: &str| seal(SECRET, r);
    added.sealed("a sealed URL", "admitted", sized, sealed);
    added.sealed("sealed under the current key ID", "admitted", sized, |r| {
        seal_keyed(CURRENT_ID, SECRET, r)
    });
    added.sealed("sealed under the previous key ID", "admitted", sized, |r| {
        seal_keyed(PREVIOUS_ID, PREVIOUS_SECRET, r)
    });
    added.sealed("sealed under the wrong key ID", "refused", sized, |r| {
        seal_keyed(PREVIOUS_ID, SECRET, r)
    });
    added.sealed("a marker that is no token", "refused", sized, |r| {
        format!("s9.notatoken/{r}")
    });
    let payloads: [SealedPayload; 7] = [
        ("sealed and expired", "refused", |_, src| {
            format!("exp:1/300x200/{src}")
        }),
        ("sealed with a grammar failure", "refused", |_, src| {
            format!("fit-in:-300x200/{src}")
        }),
        ("a sealed preset", "admitted", |_, src| {
            format!("preset:thumb/{src}")
        }),
        ("a sealed unknown preset", "refused", |_, src| {
            format!("preset:missing/{src}")
        }),
        ("sealed, the defaults past the cap", "refused", |_, src| {
            format!("{}/{src}", at_cap())
        }),
        ("a sealed root source", "admitted", |origin, src| {
            sized(&src.replace(origin, "root:origin"))
        }),
        ("a sealed purge segment", "refused", |_, src| {
            format!("purge/300x200/{src}")
        }),
    ];
    for (name, expect, payload) in payloads {
        added.sealed(name, expect, |src| payload(origin, src), sealed);
    }
}

/// What is wrong with one case's two answers, read the way
/// [`Answer::refused`] reads admission.
fn problems(case: &Case, explain: &Answer, image: &Answer) -> Vec<String> {
    let (explain_refused, image_refused) = (explain.refused(), image.refused());
    let mut problems = Vec::new();
    if case.sealed {
        // Never unsealed by `/explain`: the uniform refusal,
        // and no fetch.
        if !explain_refused || explain.code.as_deref() != Some("bad_signature") {
            problems.push("/explain did not refuse a sealed spelling".to_string());
        }
    } else if explain_refused != image_refused {
        let verdict = |refused: bool| if refused { "refuses" } else { "admits" };
        problems.push(format!(
            "/explain {} it and the image route {} it",
            verdict(explain_refused),
            verdict(image_refused)
        ));
    } else if explain_refused
        && (explain.status, &explain.code, &explain.message)
            != (image.status, &image.code, &image.message)
    {
        problems.push("the routes refuse it with different answers".to_string());
    } else if !explain_refused
        && case.marker.is_some()
        && !(explain.status == StatusCode::OK && explain.fetched)
    {
        problems.push("/explain admitted it without reporting on its source".to_string());
    }
    // The definition above reads admission off the answers; an
    // admitted request whose source is the origin's goes on to
    // fetch it, unless the meta document cannot be made at all.
    if !image_refused
        && case.marker.is_some()
        && !image.fetched
        && image.code.as_deref() != Some("meta_not_available")
    {
        problems.push("the image route admitted it and never fetched the source".to_string());
    }
    if let Some(expect) = case.expect {
        let met = match expect {
            "admitted" => !image_refused,
            "refused" => image_refused,
            code => image_refused && image.code.as_deref() == Some(code),
        };
        if !met {
            problems.push(format!("the image route should have {expect} it"));
        }
    }
    problems
}

#[tokio::test]
async fn explain_and_the_image_route_agree_on_admission() {
    let (origin, fetches) = spawn_recording_origin().await;
    let base = spawn_state(admission_state(&origin)).await;
    let json = Some("application/json");
    let mut cases = corpus_cases(&origin);
    cases.extend(added_cases(&origin));
    let mut wrong = Vec::new();
    let (mut admitted, mut refused) = (0, 0);
    for case in &cases {
        let marker = case.marker.as_deref();
        // JSON, so every refusal carries its code. On the
        // image route it asks for the meta document, which
        // runs the whole miss path; a sealed case asks for the
        // image instead, whose refusals are plain text anyway.
        let explain_path = format!("explain/{}", case.path);
        let explain = answer(&base, &explain_path, json, &fetches, marker).await;
        let image_accept = if case.sealed { None } else { json };
        let image = answer(&base, &case.path, image_accept, &fetches, marker).await;
        let problems = problems(case, &explain, &image);
        if !problems.is_empty() {
            let path: String = case.path.chars().take(160).collect();
            wrong.push(format!(
                "{} ({path}): {}\n  /explain: {explain:?}\n  image:    {image:?}",
                case.name,
                problems.join("; ")
            ));
        } else if image.refused() {
            refused += 1;
        } else {
            admitted += 1;
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases:\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
    // Neither verdict may be vacuous: the corpus alone has 156
    // URLs the grammar admits and 164 it refuses.
    assert!(
        admitted > 150 && refused > 150,
        "{admitted} admitted, {refused} refused"
    );
}
