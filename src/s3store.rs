//! Shared result storage (issue #142): the multi-replica tier
//! consulted after the local memory and disk caches, backed by
//! one S3 bucket (or an S3-compatible endpoint).
//!
//! Entries are the disk store's serialized form (the CRABDISK
//! bytes: key and cache groups embedded, MACed under a key
//! derived from the server secret), one object per entry,
//! named by the SHA-256 of the cache key under a configurable
//! prefix. Purges match on each entry's stored key and
//! groups, read from the entry header alone (a ranged read,
//! never the whole object; audit R2-24), and per-URL purges
//! additionally delete the target's enumerable key variants
//! directly by their deterministic object names.
//!
//! Like the disk store, every operation is best-effort: a
//! network or service error degrades to a miss or a dropped
//! write, never a failed request, because a cache must not
//! take the pipeline down with it. Requests are signed with
//! SigV4 exactly like the source loader's (`sigv4`), with the
//! same ambient AWS credential chain; with no credentials the
//! requests go unsigned, for local S3-compatible development
//! stores.

use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::cache::{CachedImage, key_belongs_to_source};
use crate::sigv4::{self, Credentials};
use crate::storage::{MAX_ENTRY_BYTES, key_stem, parse_entry, parse_entry_header, serialize_entry};

/// The outcome of a conditional raw write
/// ([`S3Store::put_raw_if_absent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawPut {
    /// The object was stored.
    Stored,
    /// An object already existed under the name; nothing
    /// was written.
    AlreadyExists,
    /// The store refused or could not be reached.
    Failed,
}

#[derive(Debug)]
pub struct S3Store {
    client: reqwest::Client,
    bucket: String,
    /// Normalized object prefix: empty, or ending in `/`.
    prefix: String,
    endpoint: Option<String>,
    region: String,
    creds: Option<Credentials>,
    /// The cache entry MAC key, derived from the server
    /// secret (issue #190): entries are written and
    /// verified under it, so an object planted in the
    /// bucket by anyone without the secret is a miss.
    /// Inert for the raw-object path (`put_raw`/`get_raw`
    /// store uploads verbatim, with no entry framing).
    entry_key: Vec<u8>,
    /// Largest object an entry read will buffer. Objects in
    /// the shared bucket are operator-adjacent data, not
    /// fully trusted input: a planted oversized object must
    /// not turn a lookup or a purge walk into an unbounded
    /// allocation.
    max_entry_bytes: usize,
}

impl S3Store {
    pub fn new(
        bucket: String,
        prefix: String,
        endpoint: Option<String>,
        region: String,
        creds: Option<Credentials>,
        entry_key: Vec<u8>,
    ) -> Self {
        let trimmed = prefix.trim_matches('/');
        let prefix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("{trimmed}/")
        };
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            bucket,
            prefix,
            endpoint,
            region,
            creds,
            entry_key,
            max_entry_bytes: MAX_ENTRY_BYTES,
        }
    }

    /// Read a response body under the entry size ceiling,
    /// streaming so an oversized object is refused as soon
    /// as it passes the cap instead of after it has been
    /// buffered whole. `None` on a transport error or an
    /// oversized body; both read as a cache miss upstream.
    async fn read_capped(&self, mut resp: reqwest::Response) -> Option<Vec<u8>> {
        if resp
            .content_length()
            .is_some_and(|n| n > self.max_entry_bytes as u64)
        {
            return None;
        }
        let mut data = Vec::new();
        while let Some(chunk) = resp.chunk().await.ok()? {
            if data.len().checked_add(chunk.len())? > self.max_entry_bytes {
                return None;
            }
            data.extend_from_slice(&chunk);
        }
        Some(data)
    }

    fn object_key(&self, stem: &str) -> String {
        format!("{}{stem}.bin", self.prefix)
    }

    /// One signed (or unsigned) request against the bucket.
    /// `object_key` empty addresses the bucket itself (LIST).
    /// Returns the response, or None on any transport error:
    /// callers read every failure as a cache miss.
    async fn request(
        &self,
        method: &str,
        object_key: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Option<reqwest::Response> {
        self.request_h(method, object_key, query, body, &[]).await
    }

    /// [`S3Store::request`] with extra request headers
    /// (the purge walk's `Range`).
    async fn request_h(
        &self,
        method: &str,
        object_key: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
        extra_headers: &[(&str, &str)],
    ) -> Option<reqwest::Response> {
        let path_part = if object_key.is_empty() {
            String::new()
        } else {
            format!("/{}", sigv4::encode_path(object_key))
        };
        let mut url_string = match &self.endpoint {
            Some(ep) => format!("{ep}/{}{path_part}", self.bucket),
            None => format!(
                "https://{}.s3.{}.amazonaws.com{path_part}",
                self.bucket, self.region
            ),
        };
        if !query.is_empty() {
            let qs: Vec<String> = query
                .iter()
                .map(|(k, v)| format!("{}={}", sigv4::uri_encode(k), sigv4::uri_encode(v)))
                .collect();
            url_string = format!("{url_string}?{}", qs.join("&"));
        }
        let url = reqwest::Url::parse(&url_string).ok()?;
        let payload_hash = {
            let digest = Sha256::digest(&body);
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        let verb = reqwest::Method::from_bytes(method.as_bytes()).ok()?;
        let mut request = self
            .client
            .request(verb, url.clone())
            .header("x-amz-content-sha256", payload_hash.clone())
            .body(body);
        for (name, value) in extra_headers {
            request = request.header(*name, *value);
        }
        if let Some(creds) = &self.creds {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let amz_date = sigv4::amz_date(now);
            let host = match url.port() {
                Some(port) => format!("{}:{port}", url.host_str().unwrap_or_default()),
                None => url.host_str().unwrap_or_default().to_string(),
            };
            let mut sign_headers: Vec<(&str, &str)> = vec![
                ("host", host.as_str()),
                ("x-amz-content-sha256", payload_hash.as_str()),
                ("x-amz-date", amz_date.as_str()),
            ];
            if let Some(token) = &creds.session_token {
                sign_headers.push(("x-amz-security-token", token.as_str()));
            }
            let authorization = sigv4::authorization(
                creds,
                &sigv4::SignRequest {
                    method,
                    path: url.path(),
                    query,
                    headers: &sign_headers,
                    region: &self.region,
                    service: "s3",
                    payload_sha256: &payload_hash,
                    amz_date: &amz_date,
                },
            );
            request = request
                .header("Authorization", authorization)
                .header("x-amz-date", amz_date);
            if let Some(token) = &creds.session_token {
                request = request.header("x-amz-security-token", token.as_str());
            }
        }
        request.send().await.ok()
    }

    /// Look up `key`. Any failure (missing object, unreadable
    /// entry, transport error) is a miss.
    pub async fn get(&self, key: &str) -> Option<CachedImage> {
        let object = self.object_key(&key_stem(key));
        let resp = self.request("GET", &object, &[], Vec::new()).await?;
        if !resp.status().is_success() {
            return None;
        }
        let data = self.read_capped(resp).await?;
        let (stored_key, image) = parse_entry(&data, &self.entry_key)?;
        if stored_key != key {
            return None;
        }
        Some(image)
    }

    /// Store `image` under `key`. Errors are dropped silently:
    /// the local tiers already hold the entry.
    pub async fn put(&self, key: &str, image: &CachedImage) {
        let object = self.object_key(&key_stem(key));
        let data = serialize_entry(key, image, &self.entry_key);
        let _ = self.request("PUT", &object, &[], data).await;
    }

    /// Store raw bytes under `name`, verbatim (issue #168):
    /// the object key is the store prefix plus `name`, with
    /// no cache-entry serialization and no `.bin` suffix.
    /// This is the upload destination's write path, so
    /// unlike the cache `put` it reports failure: an upload
    /// that did not land must not answer as if it did.
    /// Returns whether the store accepted the object.
    pub async fn put_raw(&self, name: &str, bytes: Vec<u8>) -> bool {
        let object = format!("{}{name}", self.prefix);
        match self.request("PUT", &object, &[], bytes).await {
            Some(resp) => resp.status().is_success(),
            None => false,
        }
    }

    /// Read the raw bytes stored under `name` (see
    /// [`S3Store::put_raw`]). Any failure (missing object,
    /// transport error) is `None`. The read half of the
    /// raw pair; the upload existence check no longer uses
    /// it (it goes through [`S3Store::exists_raw`], a
    /// HEAD, since issue #192), serving reads of uploaded
    /// objects go through the loader.
    #[allow(dead_code)]
    pub async fn get_raw(&self, name: &str) -> Option<Vec<u8>> {
        let object = format!("{}{name}", self.prefix);
        let resp = self.request("GET", &object, &[], Vec::new()).await?;
        if !resp.status().is_success() {
            return None;
        }
        resp.bytes().await.ok().map(|b| b.to_vec())
    }

    /// Whether a raw object exists under `name`, answered
    /// with a HEAD: no body is read (issue #192, audit
    /// R2-23). The upload endpoint's existence check runs
    /// on every signed PUT; the object it asks about can
    /// be arbitrarily large, and downloading it whole to
    /// answer a boolean made the check a bandwidth and
    /// memory sink.
    pub async fn exists_raw(&self, name: &str) -> bool {
        let object = format!("{}{name}", self.prefix);
        self.object_exists(&object).await
    }

    /// Store raw bytes under `name` only if no object is
    /// there yet (issue #192, audit R2-21): a conditional
    /// PUT (`If-None-Match: *`), so the existence decision
    /// and the write are one atomic step instead of a
    /// check-then-write pair another upload can slip
    /// between. Stores without conditional-write support
    /// degrade to an unconditional write; Amazon S3 and
    /// current S3-compatible stores honor the precondition
    /// and answer `412` for a name already taken.
    pub async fn put_raw_if_absent(&self, name: &str, bytes: Vec<u8>) -> RawPut {
        let object = format!("{}{name}", self.prefix);
        match self
            .request_h("PUT", &object, &[], bytes, &[("if-none-match", "*")])
            .await
        {
            Some(resp) if resp.status().is_success() => RawPut::Stored,
            Some(resp) if resp.status() == reqwest::StatusCode::PRECONDITION_FAILED => {
                RawPut::AlreadyExists
            }
            _ => RawPut::Failed,
        }
    }

    /// Every object key under the store prefix, following
    /// continuation tokens. An error yields what was listed so
    /// far (possibly nothing).
    async fn list_objects(&self) -> Vec<String> {
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query: Vec<(&str, &str)> = vec![("list-type", "2")];
            if !self.prefix.is_empty() {
                query.push(("prefix", self.prefix.as_str()));
            }
            if let Some(t) = &token {
                query.push(("continuation-token", t.as_str()));
            }
            let Some(resp) = self.request("GET", "", &query, Vec::new()).await else {
                break;
            };
            if !resp.status().is_success() {
                break;
            }
            let Ok(text) = resp.text().await else { break };
            for chunk in text.split("<Key>").skip(1) {
                if let Some(end) = chunk.find("</Key>") {
                    keys.push(xml_unescape(&chunk[..end]));
                }
            }
            if !text.contains("<IsTruncated>true</IsTruncated>") {
                break;
            }
            token = text
                .split("<NextContinuationToken>")
                .nth(1)
                .and_then(|rest| rest.split("</NextContinuationToken>").next())
                .map(xml_unescape);
            if token.is_none() {
                break;
            }
        }
        keys
    }

    /// How much of an object the purge walk reads to learn
    /// its stored key and groups: the header (field lines,
    /// the key, the groups block) sits at the front of every
    /// entry, and real keys and group lists are far smaller
    /// than this. A header that outgrows the probe falls
    /// back to one full read of that object.
    const HEADER_PROBE_BYTES: usize = 16 * 1024;

    /// Read one object's stored key and groups from its
    /// header alone, via a ranged GET (audit R2-24: a purge
    /// used to download every entry in the tier, image
    /// bytes included). The header is a match hint, not a
    /// serving read: it is parsed without MAC verification,
    /// because verification needs the whole object. A
    /// forged header can at worst steer a purge; entries
    /// are only ever SERVED through `get`, which verifies.
    /// Objects past the entry size ceiling are refused
    /// outright, exactly as `get` refuses them.
    async fn read_entry_header(&self, object: &str) -> Option<(String, Vec<String>)> {
        let range = format!("bytes=0-{}", Self::HEADER_PROBE_BYTES - 1);
        let resp = self
            .request_h("GET", object, &[], Vec::new(), &[("range", range.as_str())])
            .await?;
        if !resp.status().is_success() {
            return None;
        }
        let partial = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        // The object's real length: from Content-Range on a
        // partial response, else the response's own length.
        let total = resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit('/').next())
            .and_then(|v| v.parse::<u64>().ok())
            .or_else(|| resp.content_length())?;
        if total > self.max_entry_bytes as u64 {
            return None;
        }
        let data = self.read_capped(resp).await?;
        if !partial || data.len() as u64 >= total {
            // The whole object arrived (the store ignored
            // the range, or the entry fits in the probe).
            let (key, image) = parse_entry(&data, &self.entry_key)?;
            return Some((key, image.groups));
        }
        match parse_entry_header(&data, total as usize) {
            Some(found) => Some(found),
            None => {
                // The header outgrew the probe: one full
                // read of this object settles it.
                let resp = self.request("GET", object, &[], Vec::new()).await?;
                if !resp.status().is_success() {
                    return None;
                }
                let data = self.read_capped(resp).await?;
                let (key, image) = parse_entry(&data, &self.entry_key)?;
                Some((key, image.groups))
            }
        }
    }

    /// Whether an object exists (a HEAD, no body read).
    async fn object_exists(&self, object: &str) -> bool {
        match self.request("HEAD", object, &[], Vec::new()).await {
            Some(resp) => resp.status().is_success(),
            None => false,
        }
    }

    /// Delete one object; returns whether the store
    /// accepted the deletion.
    async fn delete_object(&self, object: &str) -> bool {
        match self.request("DELETE", object, &[], Vec::new()).await {
            Some(resp) => resp.status().is_success(),
            None => false,
        }
    }

    /// Remove every entry whose stored key is `base` or one
    /// of its `#`-suffixed variants. Returns how many
    /// entries were removed.
    ///
    /// `base` is the purge target's canonical form;
    /// `epoch_base` is the form entries are actually keyed
    /// under in the current configuration (the base plus
    /// the config fingerprint segment, audit R2-08). The
    /// enumerable variants of the epoch base (itself and
    /// the four negotiated-class keys) are deleted directly
    /// by their deterministic object names, with no listing
    /// and no reads (audit R2-24); a header walk then
    /// catches any remaining spellings (request-hint
    /// variants, older epochs) by their stored keys.
    pub async fn purge_matching(&self, base: &str, epoch_base: &str) -> usize {
        let mut removed = 0;
        for candidate in [
            epoch_base.to_string(),
            format!("{epoch_base}#avif"),
            format!("{epoch_base}#jxl"),
            format!("{epoch_base}#webp"),
            format!("{epoch_base}#source"),
        ] {
            let object = self.object_key(&key_stem(&candidate));
            if self.object_exists(&object).await && self.delete_object(&object).await {
                removed += 1;
            }
        }
        removed += self
            .purge_where(|stored_key, _| {
                stored_key == base || stored_key.starts_with(&format!("{base}#"))
            })
            .await;
        removed
    }

    /// Remove every entry belonging to `group`. Returns
    /// how many entries were removed.
    pub async fn purge_group(&self, group: &str) -> usize {
        self.purge_where(|_, groups| groups.iter().any(|g| g == group))
            .await
    }

    /// Remove every entry whose result derives from
    /// `source`, whatever operations produced it (audit
    /// R2-07): an upload overwrite calls this so the
    /// replaced original's derivatives stop serving.
    /// Returns how many entries were removed.
    pub async fn purge_source(&self, source: &str) -> usize {
        self.purge_where(|stored_key, _| key_belongs_to_source(stored_key, source))
            .await
    }

    /// The shared purge walk: list, read each entry's
    /// header, delete the matches. Reads are header-sized
    /// (see [`S3Store::read_entry_header`]), never whole
    /// entries.
    async fn purge_where(&self, matches: impl Fn(&str, &[String]) -> bool) -> usize {
        let mut removed = 0;
        for object in self.list_objects().await {
            let Some((stored_key, groups)) = self.read_entry_header(&object).await else {
                continue;
            };
            if !matches(&stored_key, &groups) {
                continue;
            }
            if self.delete_object(&object).await {
                removed += 1;
            }
        }
        removed
    }
}

/// The five predefined XML entities, the only escaping S3
/// applies to listed keys.
fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex as StdMutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Injectable failure switches for the error-path
    /// tests: while a flag is set, the matching operation
    /// answers 500 instead of doing its work.
    #[derive(Default)]
    pub struct FailFlags {
        pub get_object: bool,
        pub put: bool,
        pub delete: bool,
        pub list: bool,
    }

    /// A minimal in-process S3: PUT/GET/DELETE plus
    /// ListObjectsV2 over one shared map, path-style URLs.
    pub struct FakeS3 {
        pub objects: Arc<StdMutex<HashMap<String, Vec<u8>>>>,
        pub fail: Arc<StdMutex<FailFlags>>,
        /// One line per request served, `"<METHOD> <key>"`
        /// (plus the conditional-write marker when the
        /// request carried one), so tests can assert how a
        /// lookup was answered, not just what it returned.
        pub requests: Arc<StdMutex<Vec<String>>>,
        pub endpoint: String,
    }

    pub async fn start_fake() -> FakeS3 {
        let objects = Arc::new(StdMutex::new(HashMap::new()));
        let fail = Arc::new(StdMutex::new(FailFlags::default()));
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let shared = objects.clone();
        let shared_fail = fail.clone();
        let shared_requests = requests.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let objects = shared.clone();
                let fail = shared_fail.clone();
                let requests = shared_requests.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 8192];
                    let header_end;
                    loop {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(pos) = find(&buf, b"\r\n\r\n") {
                            header_end = pos + 4;
                            break;
                        }
                    }
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    let mut lines = head.lines();
                    let request_line = lines.next().unwrap_or_default().to_string();
                    let mut content_length: usize = 0;
                    let mut range_end: Option<usize> = None;
                    let mut if_none_match: Option<String> = None;
                    for line in lines {
                        let lower = line.to_ascii_lowercase();
                        if let Some(v) = lower.strip_prefix("content-length:") {
                            content_length = v.trim().parse().unwrap_or(0);
                        }
                        if let Some(v) = lower.strip_prefix("range:") {
                            range_end = v
                                .trim()
                                .strip_prefix("bytes=0-")
                                .and_then(|n| n.parse().ok());
                        }
                        if let Some(v) = lower.strip_prefix("if-none-match:") {
                            if_none_match = Some(v.trim().to_string());
                        }
                    }
                    while buf.len() < header_end + content_length {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = buf[header_end..header_end + content_length].to_vec();
                    let mut parts = request_line.split_whitespace();
                    let method = parts.next().unwrap_or_default().to_string();
                    let target = parts.next().unwrap_or_default().to_string();
                    {
                        let (path, _) = match target.split_once('?') {
                            Some((p, q)) => (p, q),
                            None => (target.as_str(), ""),
                        };
                        let rest = path.trim_start_matches('/');
                        let key =
                            percent_decode(rest.split_once('/').map(|(_, k)| k).unwrap_or(""));
                        let mut line = format!("{method} {key}");
                        if if_none_match.is_some() {
                            line.push_str(" if-none-match");
                        }
                        requests.lock().unwrap().push(line);
                    }
                    let (status, payload, extra) = handle(
                        &objects,
                        &fail,
                        &method,
                        &target,
                        body,
                        range_end,
                        if_none_match.as_deref(),
                    );
                    let resp = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: {}\r\n{extra}connection: close\r\n\r\n",
                        payload.len()
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    // A HEAD response advertises the object's
                    // length but carries no body.
                    if method != "HEAD" {
                        let _ = socket.write_all(&payload).await;
                    }
                });
            }
        });
        FakeS3 {
            objects,
            fail,
            requests,
            endpoint: format!("http://127.0.0.1:{port}"),
        }
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    fn percent_decode(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).to_string()
    }

    fn handle(
        objects: &StdMutex<HashMap<String, Vec<u8>>>,
        fail: &StdMutex<FailFlags>,
        method: &str,
        target: &str,
        body: Vec<u8>,
        range_end: Option<usize>,
        if_none_match: Option<&str>,
    ) -> (String, Vec<u8>, String) {
        let (path, query) = match target.split_once('?') {
            Some((p, q)) => (p, q),
            None => (target, ""),
        };
        let rest = path.trim_start_matches('/');
        let key = percent_decode(rest.split_once('/').map(|(_, k)| k).unwrap_or(""));
        let fail = fail.lock().unwrap();
        let error = (
            "500 Internal Server Error".into(),
            Vec::new(),
            String::new(),
        );
        match (method, query.contains("list-type=2")) {
            ("PUT", _) if fail.put => error,
            ("PUT", _)
                if if_none_match == Some("*") && objects.lock().unwrap().contains_key(&key) =>
            {
                // The conditional write's precondition: a
                // name already taken is a 412, nothing is
                // written.
                ("412 Precondition Failed".into(), Vec::new(), String::new())
            }
            ("PUT", _) => {
                objects.lock().unwrap().insert(key, body);
                ("200 OK".into(), Vec::new(), String::new())
            }
            ("GET", true) if fail.list => error,
            ("GET", true) => {
                let prefix = query
                    .split('&')
                    .find_map(|p| p.strip_prefix("prefix="))
                    .map(percent_decode)
                    .unwrap_or_default();
                let mut xml = String::from(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><IsTruncated>false</IsTruncated>",
                );
                for k in objects.lock().unwrap().keys() {
                    if k.starts_with(&prefix) {
                        xml += &format!("<Contents><Key>{k}</Key></Contents>");
                    }
                }
                xml += "</ListBucketResult>";
                ("200 OK".into(), xml.into_bytes(), String::new())
            }
            ("GET", false) if fail.get_object => error,
            ("GET", false) => match objects.lock().unwrap().get(&key) {
                Some(v) => match range_end {
                    Some(end) if end + 1 < v.len() => {
                        let total = v.len();
                        (
                            "206 Partial Content".into(),
                            v[..=end].to_vec(),
                            format!("content-range: bytes 0-{end}/{total}\r\n"),
                        )
                    }
                    _ => ("200 OK".into(), v.clone(), String::new()),
                },
                None => ("404 Not Found".into(), Vec::new(), String::new()),
            },
            ("HEAD", _) if fail.get_object => error,
            ("HEAD", _) => match objects.lock().unwrap().get(&key) {
                // The payload stands in for the length; the
                // connection loop writes none of it.
                Some(v) => ("200 OK".into(), v.clone(), String::new()),
                None => ("404 Not Found".into(), Vec::new(), String::new()),
            },
            ("DELETE", _) if fail.delete => error,
            ("DELETE", _) => {
                objects.lock().unwrap().remove(&key);
                ("204 No Content".into(), Vec::new(), String::new())
            }
            _ => ("400 Bad Request".into(), Vec::new(), String::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::start_fake;
    use super::*;
    use crate::url::ImageFormat;

    /// The entry MAC key the test stores run under.
    fn test_entry_key() -> Vec<u8> {
        crate::url::entry_key(b"s3store-test-secret")
    }

    fn store(endpoint: &str) -> S3Store {
        S3Store::new(
            "results".into(),
            "cache/".into(),
            Some(endpoint.into()),
            "us-east-1".into(),
            None,
            test_entry_key(),
        )
    }

    fn image(bytes: &[u8], groups: &[&str]) -> CachedImage {
        CachedImage {
            bytes: bytes.to_vec(),
            format: ImageFormat::WebP,
            quality: Some(80),
            groups: groups.iter().map(|g| g.to_string()).collect(),
        }
    }

    #[tokio::test]
    async fn put_get_round_trip_across_instances() {
        let fake = start_fake().await;
        let a = store(&fake.endpoint);
        let b = store(&fake.endpoint);
        assert!(a.get("300x200/example.com/i.jpg").await.is_none());
        a.put("300x200/example.com/i.jpg", &image(b"pixels", &["hero"]))
            .await;
        // A second instance (another replica) reads it back.
        let got = b.get("300x200/example.com/i.jpg").await.unwrap();
        assert_eq!(got.bytes, b"pixels");
        assert_eq!(got.groups, vec!["hero".to_string()]);
        assert_eq!(got.quality, Some(80));
    }

    #[tokio::test]
    async fn purge_matching_removes_the_base_and_its_variants_only() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        for key in [
            "a/example.com/i.jpg",
            "a/example.com/i.jpg#w2",
            "ab/example.com/i.jpg",
        ] {
            s.put(key, &image(b"x", &[])).await;
        }
        assert_eq!(
            s.purge_matching("a/example.com/i.jpg", "a/example.com/i.jpg")
                .await,
            2
        );
        assert!(s.get("a/example.com/i.jpg").await.is_none());
        assert!(s.get("ab/example.com/i.jpg").await.is_some());
    }

    #[tokio::test]
    async fn purge_group_removes_only_its_members() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("one", &image(b"1", &["hero"])).await;
        s.put("two", &image(b"2", &["hero", "side"])).await;
        s.put("three", &image(b"3", &["side"])).await;
        assert_eq!(s.purge_group("hero").await, 2);
        assert!(s.get("three").await.is_some());
        assert_eq!(s.purge_group("hero").await, 0);
    }

    fn store_with_prefix(endpoint: &str, prefix: &str) -> S3Store {
        S3Store::new(
            "results".into(),
            prefix.into(),
            Some(endpoint.into()),
            "us-east-1".into(),
            None,
            test_entry_key(),
        )
    }

    #[tokio::test]
    async fn read_errors_and_corrupt_entries_degrade_to_misses() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("key", &image(b"pixels", &[])).await;
        assert!(s.get("key").await.is_some());
        // A service error on the read is a miss, never a
        // failed lookup: the local tiers and the origin
        // remain available to the request.
        fake.fail.lock().unwrap().get_object = true;
        assert!(s.get("key").await.is_none());
        fake.fail.lock().unwrap().get_object = false;
        assert!(s.get("key").await.is_some());
        // An object whose bytes are not a serialized entry
        // reads as a miss too.
        fake.objects.lock().unwrap().insert(
            format!("cache/{}.bin", key_stem("key")),
            b"not an entry".to_vec(),
        );
        assert!(s.get("key").await.is_none());
    }

    #[tokio::test]
    async fn oversized_and_poisoned_objects_degrade_to_misses() {
        let fake = start_fake().await;
        let mut s = store(&fake.endpoint);
        s.max_entry_bytes = 128;
        // Under the ceiling, entries round trip as before.
        s.put("small", &image(b"x", &["hero"])).await;
        assert!(s.get("small").await.is_some());
        // Over the ceiling, the read refuses the object
        // instead of buffering it whole (R2-19), and the
        // purge walk skips it the same way: only the
        // readable entry is removed.
        s.put("big", &image(&vec![7u8; 500], &["hero"])).await;
        assert!(s.get("big").await.is_none());
        assert_eq!(s.purge_group("hero").await, 1);
        // A header claiming a group count far beyond the
        // object is a miss, never a panic or an allocation
        // sized by the count (R2-13b).
        fake.objects.lock().unwrap().insert(
            format!("cache/{}.bin", key_stem("evil")),
            b"CRABDISK4\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nwebp\n80\n999999999999999999\n4\n1\nevil\nx".to_vec(),
        );
        assert!(s.get("evil").await.is_none());
        // A validly framed entry with the wrong MAC (a
        // plant by someone without the secret, or an object
        // written under another secret) is a miss too
        // (R2-13 part a).
        let forged = {
            let mut data = crate::storage::serialize_entry(
                "planted",
                &image(b"forged", &[]),
                &test_entry_key(),
            );
            let last = data.len() - 1;
            data[last] ^= 0xFF;
            data
        };
        fake.objects
            .lock()
            .unwrap()
            .insert(format!("cache/{}.bin", key_stem("planted")), forged);
        assert!(s.get("planted").await.is_none());
    }

    #[tokio::test]
    async fn write_errors_are_dropped_silently() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        fake.fail.lock().unwrap().put = true;
        // The write reports nothing and stores nothing: the
        // failed put must not surface as an error (put has
        // no failure channel) nor leave a partial entry.
        s.put("key", &image(b"pixels", &[])).await;
        fake.fail.lock().unwrap().put = false;
        assert!(s.get("key").await.is_none());
        assert!(fake.objects.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn purge_errors_remove_nothing_and_report_zero() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("one", &image(b"1", &["hero"])).await;
        s.put("two", &image(b"2", &["hero"])).await;
        // A listing failure truncates the walk: nothing is
        // read, nothing deleted, count zero.
        fake.fail.lock().unwrap().list = true;
        assert_eq!(s.purge_group("hero").await, 0);
        // A per-URL purge no longer depends on the listing
        // for the enumerable variants: the epoch base
        // deletes directly even while the walk is
        // truncated (R2-24).
        assert_eq!(s.purge_matching("one", "one").await, 1);
        assert!(s.get("one").await.is_none());
        s.put("one", &image(b"1", &["hero"])).await;
        fake.fail.lock().unwrap().list = false;
        // Entry reads failing mid-walk skip those entries
        // the same way.
        fake.fail.lock().unwrap().get_object = true;
        assert_eq!(s.purge_group("hero").await, 0);
        fake.fail.lock().unwrap().get_object = false;
        // Deletes failing leave every entry in place, and
        // only successful deletes count once they recover.
        fake.fail.lock().unwrap().delete = true;
        assert_eq!(s.purge_group("hero").await, 0);
        fake.fail.lock().unwrap().delete = false;
        assert!(s.get("one").await.is_some());
        assert_eq!(s.purge_group("hero").await, 2);
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_degrades_like_a_failing_bucket() {
        // A port nothing listens on: every request fails at
        // the transport level instead of the service level.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let s = store(&format!("http://127.0.0.1:{port}"));
        assert!(s.get("key").await.is_none());
        s.put("key", &image(b"pixels", &[])).await;
        assert_eq!(s.purge_group("hero").await, 0);
        assert_eq!(s.purge_matching("key", "key").await, 0);
    }

    #[tokio::test]
    async fn prefixes_namespace_the_shared_bucket() {
        let fake = start_fake().await;
        let a = store_with_prefix(&fake.endpoint, "a/");
        let b = store_with_prefix(&fake.endpoint, "b/");
        a.put("same-key", &image(b"from-a", &["hero"])).await;
        // The same cache key under another prefix is a
        // different entry: invisible, and independently held.
        assert!(b.get("same-key").await.is_none());
        b.put("same-key", &image(b"from-b", &[])).await;
        assert_eq!(a.get("same-key").await.unwrap().bytes, b"from-a");
        assert_eq!(b.get("same-key").await.unwrap().bytes, b"from-b");
        assert!(
            fake.objects
                .lock()
                .unwrap()
                .keys()
                .all(|k| k.starts_with("a/") || k.starts_with("b/")),
            "objects land under their store's prefix"
        );
        // A purge under one prefix never reaches the other.
        assert_eq!(a.purge_group("hero").await, 1);
        assert!(a.get("same-key").await.is_none());
        assert_eq!(b.get("same-key").await.unwrap().bytes, b"from-b");
        assert_eq!(b.purge_matching("same-key", "same-key").await, 1);
        assert!(fake.objects.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn purge_source_removes_every_derivative_of_the_source() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        s.put("300x200/file:///heroes/cat.jpg", &image(b"1", &[]))
            .await;
        s.put(
            "300x200/file:///heroes/cat.jpg#cfgab12#avif",
            &image(b"2", &[]),
        )
        .await;
        s.put("600x400/file:///heroes/cat.jpg", &image(b"3", &[]))
            .await;
        s.put("300x200/file:///heroes/cat.jpg2", &image(b"4", &[]))
            .await;
        s.put("300x200/file:///heroes/dog.jpg", &image(b"5", &[]))
            .await;
        assert_eq!(s.purge_source("file:///heroes/cat.jpg").await, 3);
        assert!(s.get("300x200/file:///heroes/cat.jpg").await.is_none());
        assert!(s.get("300x200/file:///heroes/cat.jpg2").await.is_some());
        assert!(s.get("300x200/file:///heroes/dog.jpg").await.is_some());
    }

    #[tokio::test]
    async fn purge_matches_on_the_header_without_the_payload() {
        let fake = start_fake().await;
        let s = store(&fake.endpoint);
        // A payload far past the 16 KB header probe: the
        // entry is larger than the probe response, so the
        // walk can only have matched on the header (R2-24).
        let payload = vec![7u8; 64 * 1024];
        s.put("300x200/http://x/big.png", &image(&payload, &["hero"]))
            .await;
        let object_len = fake.objects.lock().unwrap().values().next().unwrap().len();
        assert!(
            object_len > 16 * 1024,
            "entry outgrows the probe: {object_len}"
        );
        assert_eq!(s.purge_group("hero").await, 1);
        assert!(s.get("300x200/http://x/big.png").await.is_none());
    }

    #[tokio::test]
    async fn raw_existence_is_a_head_and_raw_puts_can_be_conditional() {
        let fake = start_fake().await;
        let s = store_with_prefix(&fake.endpoint, "orig/");
        // Existence is answered with a HEAD, never a GET:
        // the object behind the name is not downloaded to
        // answer a boolean (issue #192, audit R2-23).
        assert!(!s.exists_raw("heroes/cat.jpg").await);
        assert!(s.put_raw("heroes/cat.jpg", b"pixels".to_vec()).await);
        assert!(s.exists_raw("heroes/cat.jpg").await);
        let requests = fake.requests.lock().unwrap().clone();
        assert!(
            requests
                .iter()
                .filter(|r| r.as_str() == "HEAD orig/heroes/cat.jpg")
                .count()
                == 2,
            "{requests:?}"
        );
        assert!(
            !requests.iter().any(|r| r.starts_with("GET ")),
            "{requests:?}"
        );
        // A conditional put stores the first writer and
        // refuses the second without touching its bytes
        // (audit R2-21).
        assert_eq!(
            s.put_raw_if_absent("heroes/dog.jpg", b"one".to_vec()).await,
            RawPut::Stored
        );
        assert_eq!(
            s.put_raw_if_absent("heroes/dog.jpg", b"two".to_vec()).await,
            RawPut::AlreadyExists
        );
        assert_eq!(
            s.get_raw("heroes/dog.jpg").await.as_deref(),
            Some(b"one".as_slice())
        );
        // A store failure reads as Failed, not as a store
        // or a conflict.
        fake.fail.lock().unwrap().put = true;
        assert_eq!(
            s.put_raw_if_absent("heroes/eel.jpg", b"x".to_vec()).await,
            RawPut::Failed
        );
    }
}
