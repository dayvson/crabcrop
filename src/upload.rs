//! Uploaded originals (issue #168): the write side of the
//! source store.
//!
//! crabcrop is fetch-only by design; this module is the
//! one deliberate exception, and it is shaped to stay
//! narrow. Uploads exist only when the operator configures
//! a destination (`CRABCROP_UPLOAD_DIR` or
//! `CRABCROP_UPLOAD_BUCKET`), every upload is signed in
//! its own signature domain (`crate::url::verify_upload`),
//! the bytes must decode as an image the pipeline can
//! serve before they are stored, and there is no listing
//! and no delete here: the surface is exactly one `PUT`
//! per object.
//!
//! Two destination backends, reusing the store
//! configuration the codebase already has:
//!
//! - A local directory. Its objects are served back as
//!   `file://` sources; boot points the loader's file
//!   root at the directory when no root is configured
//!   (and refuses a directory outside a configured root).
//! - An S3 bucket (or S3-compatible endpoint), reusing
//!   the source S3 configuration (endpoint, region,
//!   ambient AWS credentials) exactly like the shared
//!   result store does. Its objects are served back as
//!   `s3://` sources.
//!
//! There is no GCS destination: `gs://` sources are
//! read-only in this codebase (the loader fetches them
//! through the XML API, but no GCS write store exists),
//! and this feature does not invent one.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::s3store::S3Store;

/// The longest object name, in bytes. Names travel in a
/// URL path and name files and object keys; 255 is the
/// common per-component filesystem limit, applied to the
/// whole name so one name is always one safe component
/// sequence on every backend.
pub const MAX_NAME_BYTES: usize = 255;

/// Validate an upload object name. The name is the PUT
/// path after the signature segment, and later the tail
/// of a `file://` or `s3://` source, so it is held to a
/// conservative shape: `/`-separated segments of
/// `[A-Za-z0-9._-]`, no empty segment (no leading,
/// trailing, or doubled slash), no `.` or `..` segment,
/// and no segment starting with `.` (no dotfiles, no
/// traversal spelled with dots). That charset excludes
/// backslashes, percent escapes, whitespace, and
/// non-ASCII outright, so the name that was signed is
/// the name that is stored, byte for byte.
pub fn validate_upload_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("upload name is empty".to_string());
    }
    if name.len() > MAX_NAME_BYTES {
        return Err(format!(
            "upload name exceeds the {MAX_NAME_BYTES}-byte limit"
        ));
    }
    for segment in name.split('/') {
        if segment.is_empty() {
            return Err(format!(
                "invalid upload name {name:?}: empty segment (leading, trailing, or doubled '/')"
            ));
        }
        if segment.starts_with('.') {
            return Err(format!(
                "invalid upload name {name:?}: segment {segment:?} must not start with '.'"
            ));
        }
        if let Some(c) = segment
            .chars()
            .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
        {
            return Err(format!(
                "invalid upload name {name:?}: character {c:?} is not allowed (segments are [A-Za-z0-9._-])"
            ));
        }
    }
    Ok(())
}

/// Why storing an upload failed, once its bytes were
/// accepted. The handler maps these: a name already taken
/// is a conflict (409), a backend that is unreachable is
/// an upstream fault (502), a local IO failure is a
/// server fault (500).
#[derive(Debug)]
pub enum UploadError {
    /// A conditional create found the name already taken.
    AlreadyExists,
    /// The S3 backend refused or could not be reached.
    Backend(String),
    /// The local directory could not be written.
    Io(String),
}

impl std::fmt::Display for UploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UploadError::AlreadyExists => {
                write!(f, "an object with this name already exists")
            }
            UploadError::Backend(cause) => write!(f, "upload store failed: {cause}"),
            UploadError::Io(cause) => write!(f, "cannot write the upload: {cause}"),
        }
    }
}

impl std::error::Error for UploadError {}

/// Where uploaded objects land.
#[derive(Debug)]
enum Backend {
    /// A canonicalized local directory; object `a/b.jpg`
    /// is the file `<dir>/a/b.jpg` under it.
    Local(PathBuf),
    /// An S3 bucket, written through the raw-object path
    /// of the shared store implementation. The bucket
    /// name is kept here for source-reference building
    /// (the store itself keeps its prefix private).
    S3 {
        store: Arc<S3Store>,
        bucket: String,
        /// Normalized prefix: empty, or ending in `/`.
        prefix: String,
    },
}

/// The configured upload destination plus its bounds.
/// Built at boot (`crate::build_upload_store`); `None`
/// on the server state means uploads do not exist on
/// this server and the route is never registered.
#[derive(Debug)]
pub struct UploadStore {
    backend: Backend,
    /// Maximum accepted body size, in bytes. Bounded at
    /// boot by the loader's source cap, so a stored
    /// original is always one the loader will fetch back.
    pub max_bytes: u64,
    /// Whether a PUT may replace an existing object.
    /// Off by default: an existing name is a 409.
    pub allow_overwrite: bool,
    /// For a local destination, the upload directory's
    /// path relative to the loader's file root, empty or
    /// ending in `/` (issue #265): `source_ref` must name
    /// the object where the loader will actually look for
    /// it, which is under the root, not at its top.
    ref_prefix: String,
}

impl UploadStore {
    /// A local-directory destination. `dir` must already
    /// be canonicalized (boot does it, like the file
    /// source root).
    pub fn local(dir: PathBuf, max_bytes: u64, allow_overwrite: bool) -> Self {
        Self {
            backend: Backend::Local(dir),
            max_bytes,
            allow_overwrite,
            ref_prefix: String::new(),
        }
    }

    /// Set the local destination's path relative to the
    /// loader's file root (issue #265): boot computes it
    /// when a root is configured above the upload dir.
    pub fn with_ref_prefix(mut self, ref_prefix: String) -> Self {
        self.ref_prefix = ref_prefix;
        self
    }

    /// An S3 destination, reusing the source S3
    /// configuration (endpoint, region, credentials).
    pub fn s3(
        bucket: String,
        prefix: String,
        endpoint: Option<String>,
        region: String,
        creds: Option<crate::sigv4::Credentials>,
        max_bytes: u64,
        allow_overwrite: bool,
    ) -> Self {
        let trimmed = prefix.trim_matches('/');
        let prefix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("{trimmed}/")
        };
        // The upload path moves raw objects only (put_raw
        // and get_raw store bytes verbatim, with no entry
        // framing), so the entry MAC key is never used
        // here; an empty one keeps that explicit.
        let store = S3Store::new(
            bucket.clone(),
            prefix.clone(),
            endpoint,
            region,
            creds,
            Vec::new(),
        );
        Self {
            backend: Backend::S3 {
                store: Arc::new(store),
                bucket,
                prefix,
            },
            max_bytes,
            allow_overwrite,
            ref_prefix: String::new(),
        }
    }

    /// The local destination directory, when the backend
    /// is one: boot points the loader's file root at it
    /// (when no root is configured) so uploaded objects
    /// serve back as `file://` sources.
    pub fn local_dir(&self) -> Option<&std::path::Path> {
        match &self.backend {
            Backend::Local(dir) => Some(dir),
            Backend::S3 { .. } => None,
        }
    }

    /// The exact source reference a caller embeds in
    /// transform URLs for the object stored under `name`:
    /// a `file://` path for a local destination (served
    /// under the loader's file root, which boot points
    /// at the upload directory), an `s3://` locator for a
    /// bucket destination.
    pub fn source_ref(&self, name: &str) -> String {
        match &self.backend {
            Backend::Local(_) => format!("file:///{}{name}", self.ref_prefix),
            Backend::S3 { bucket, prefix, .. } => format!("s3://{bucket}/{prefix}{name}"),
        }
    }

    /// Whether an object already exists under `name`.
    /// Backend errors read as "does not exist" for S3
    /// (its store contract degrades failures to misses);
    /// the write that follows surfaces a real failure.
    /// The S3 answer is a HEAD, never a download of the
    /// existing object (issue #192, audit R2-23).
    pub async fn exists(&self, name: &str) -> bool {
        match &self.backend {
            Backend::Local(dir) => dir.join(name).is_file(),
            Backend::S3 { store, .. } => store.exists_raw(name).await,
        }
    }

    /// Store `bytes` under `name`, replacing any existing
    /// object (the handler enforces the overwrite policy
    /// through [`UploadStore::exists`] before calling
    /// this). Local writes go to a temp file renamed into
    /// place, so a crash never leaves a partial object.
    pub async fn put(&self, name: &str, bytes: &[u8]) -> Result<(), UploadError> {
        match &self.backend {
            Backend::Local(dir) => {
                let dir = dir.clone();
                let name = name.to_string();
                let bytes = bytes.to_vec();
                tokio::task::spawn_blocking(move || put_local(&dir, &name, &bytes, true))
                    .await
                    .map_err(|e| UploadError::Io(format!("upload write task failed: {e}")))?
            }
            Backend::S3 { store, .. } => {
                if store.put_raw(name, bytes.to_vec()).await {
                    Ok(())
                } else {
                    Err(UploadError::Backend(
                        "the object store did not accept the upload".to_string(),
                    ))
                }
            }
        }
    }

    /// Store `bytes` under `name` only if the name is
    /// still free, atomically (issue #192, audit R2-21):
    /// the existence decision and the write are one step,
    /// so two uploads racing for one new name cannot both
    /// succeed, and the loser is answered
    /// [`UploadError::AlreadyExists`] with the winner's
    /// bytes untouched. This is the write the handler
    /// uses whenever overwriting is not allowed; the
    /// earlier [`UploadStore::exists`] check is only the
    /// cheap refusal that keeps a doomed upload from
    /// paying for a validation decode.
    pub async fn put_new(&self, name: &str, bytes: &[u8]) -> Result<(), UploadError> {
        match &self.backend {
            Backend::Local(dir) => {
                let dir = dir.clone();
                let name = name.to_string();
                let bytes = bytes.to_vec();
                tokio::task::spawn_blocking(move || put_local(&dir, &name, &bytes, false))
                    .await
                    .map_err(|e| UploadError::Io(format!("upload write task failed: {e}")))?
            }
            Backend::S3 { store, .. } => {
                match store.put_raw_if_absent(name, bytes.to_vec()).await {
                    crate::s3store::RawPut::Stored => Ok(()),
                    crate::s3store::RawPut::AlreadyExists => Err(UploadError::AlreadyExists),
                    crate::s3store::RawPut::Failed => Err(UploadError::Backend(
                        "the object store did not accept the upload".to_string(),
                    )),
                }
            }
        }
    }
}

/// Bind the object name to the media kind the body
/// validated as (issue #192, audit R2-22): when the
/// name's final segment carries an extension, the
/// extension must be one of the decoded kind's spellings.
/// Uploads store originals that an operator may serve
/// straight from the destination, so a name is not free
/// to present the bytes as something they are not: a PNG
/// cannot land as `promo.html`, and a JPEG cannot land as
/// `photo.png`. A name with no extension has nothing to
/// bind and is accepted; media kinds outside the binding
/// table ([`UploadMedia::Other`]) are likewise unbound.
pub fn check_upload_media(name: &str, media: crate::engine::UploadMedia) -> Result<(), String> {
    let Some(allowed) = media.allowed_extensions() else {
        return Ok(());
    };
    let last = name.rsplit('/').next().unwrap_or(name);
    let Some((_, ext)) = last.rsplit_once('.') else {
        return Ok(());
    };
    if ext.is_empty() {
        return Ok(());
    }
    let ext = ext.to_ascii_lowercase();
    if allowed.contains(&ext.as_str()) {
        return Ok(());
    }
    Err(format!(
        "upload is a {} image but the name ends in .{ext}; rename it to match its content ({})",
        media.label(),
        allowed
            .iter()
            .map(|e| format!(".{e}"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The temp file a local `put` writes before renaming
/// into place: process id plus a per-process counter,
/// the disk result store's scheme, so concurrent puts
/// never share one temp file.
fn tmp_name() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(".upload-tmp-{}-{n}", std::process::id())
}

/// Write one local object. With `overwrite`, the temp
/// file is renamed over the target; without it, the temp
/// file is hard-linked into place, which fails if the
/// target already exists: the create is atomic, the local
/// half of the conditional-create guarantee (issue #192,
/// audit R2-21).
///
/// Both shapes first prove the write stays inside the
/// upload directory (audit R2-25): the name itself cannot
/// traverse (it is validated), but a pre-existing
/// directory component can be a symlink planted in the
/// tree, and a write would follow it out. The parent is
/// canonicalized after creation and must sit under the
/// (canonicalized) upload root, mirroring the loader's
/// read-side check in `load_file`.
fn put_local(
    dir: &std::path::Path,
    name: &str,
    bytes: &[u8],
    overwrite: bool,
) -> Result<(), UploadError> {
    let target = dir.join(name);
    let parent = target
        .parent()
        .ok_or_else(|| UploadError::Io("the upload path has no parent".to_string()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| UploadError::Io(format!("cannot create the upload directory: {e}")))?;
    let root = std::fs::canonicalize(dir)
        .map_err(|e| UploadError::Io(format!("cannot resolve the upload directory: {e}")))?;
    let real_parent = std::fs::canonicalize(parent)
        .map_err(|e| UploadError::Io(format!("cannot resolve the upload directory: {e}")))?;
    if !real_parent.starts_with(&root) {
        return Err(UploadError::Io(format!(
            "the upload path for {name:?} escapes the upload directory"
        )));
    }
    let tmp = target.with_file_name(tmp_name());
    std::fs::write(&tmp, bytes)
        .map_err(|e| UploadError::Io(format!("cannot write the upload: {e}")))?;
    if overwrite {
        if std::fs::rename(&tmp, &target).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return Err(UploadError::Io(
                "cannot move the upload into place".to_string(),
            ));
        }
        return Ok(());
    }
    match std::fs::hard_link(&tmp, &target) {
        Ok(()) => {
            let _ = std::fs::remove_file(&tmp);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = std::fs::remove_file(&tmp);
            Err(UploadError::AlreadyExists)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(UploadError::Io(format!(
                "cannot move the upload into place: {e}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_held_to_the_conservative_charset() {
        for ok in [
            "cat.jpg",
            "heroes/cat.jpg",
            "a/b/c-d_e.F2.webp",
            "IMG_0001.avif",
        ] {
            assert!(validate_upload_name(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "/cat.jpg",
            "cat.jpg/",
            "a//b.jpg",
            "../secret.jpg",
            "a/../b.jpg",
            "a/./b.jpg",
            ".hidden.jpg",
            "a/.hidden/b.jpg",
            "a\\b.jpg",
            "a b.jpg",
            "a%20b.jpg",
            "café.jpg",
            "a:b.jpg",
            "a?b.jpg",
        ] {
            assert!(validate_upload_name(bad).is_err(), "{bad}");
        }
        let too_long = "a".repeat(MAX_NAME_BYTES + 1);
        assert!(validate_upload_name(&too_long).is_err());
        let at_limit = "a".repeat(MAX_NAME_BYTES);
        assert!(validate_upload_name(&at_limit).is_ok());
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("crabcrop-upload-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::canonicalize(&dir).unwrap()
    }

    #[tokio::test]
    async fn local_store_round_trips_and_names_source_refs() {
        let dir = temp_dir("local");
        let store = UploadStore::local(dir.clone(), 1024, false);
        assert_eq!(store.source_ref("heroes/cat.jpg"), "file:///heroes/cat.jpg");
        // With the upload dir nested under the file root,
        // the ref carries the nesting (issue #265).
        let nested =
            UploadStore::local(dir.clone(), 1024, false).with_ref_prefix("uploads/".into());
        assert_eq!(
            nested.source_ref("heroes/cat.jpg"),
            "file:///uploads/heroes/cat.jpg"
        );
        assert!(!store.exists("heroes/cat.jpg").await);
        store.put("heroes/cat.jpg", b"pixels").await.unwrap();
        assert!(store.exists("heroes/cat.jpg").await);
        assert_eq!(
            std::fs::read(dir.join("heroes/cat.jpg")).unwrap(),
            b"pixels"
        );
        store.put("heroes/cat.jpg", b"newer").await.unwrap();
        assert_eq!(std::fs::read(dir.join("heroes/cat.jpg")).unwrap(), b"newer");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn local_put_new_creates_exactly_once() {
        let dir = temp_dir("local-new");
        let store = UploadStore::local(dir.clone(), 1024, false);
        store.put_new("heroes/cat.jpg", b"first").await.unwrap();
        // The conditional create refuses the second writer
        // and leaves the winner's bytes untouched: the
        // check-then-write race is closed (audit R2-21).
        let err = store.put_new("heroes/cat.jpg", b"second").await;
        assert!(matches!(err, Err(UploadError::AlreadyExists)), "{err:?}");
        assert_eq!(std::fs::read(dir.join("heroes/cat.jpg")).unwrap(), b"first");
        // No temp files are left behind by either attempt.
        let names: Vec<String> = std::fs::read_dir(dir.join("heroes"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, ["cat.jpg"]);
        // The unconditional write still replaces.
        store.put("heroes/cat.jpg", b"replaced").await.unwrap();
        assert_eq!(
            std::fs::read(dir.join("heroes/cat.jpg")).unwrap(),
            b"replaced"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_writes_refuse_a_symlinked_escape() {
        // A directory component inside the upload tree is
        // a symlink to a directory outside it: a write for
        // a name under it must be refused, and nothing may
        // land outside (audit R2-25).
        let dir = temp_dir("local-escape");
        let outside = temp_dir("local-outside");
        std::os::unix::fs::symlink(&outside, dir.join("linked")).unwrap();
        let store = UploadStore::local(dir.clone(), 1024, true);
        let err = store.put("linked/evil.jpg", b"pixels").await;
        assert!(matches!(err, Err(UploadError::Io(_))), "{err:?}");
        assert!(!outside.join("evil.jpg").exists());
        let err = store.put_new("linked/evil.jpg", b"pixels").await;
        assert!(matches!(err, Err(UploadError::Io(_))), "{err:?}");
        assert!(!outside.join("evil.jpg").exists());
        // A name that stays inside still writes.
        store.put("plain/ok.jpg", b"pixels").await.unwrap();
        assert!(dir.join("plain/ok.jpg").exists());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn names_are_bound_to_the_validated_media() {
        use crate::engine::UploadMedia::*;
        // Matching spellings, case-insensitively, including
        // the multi-spelling kinds.
        for (name, media) in [
            ("cat.jpg", Jpeg),
            ("cat.JPEG", Jpeg),
            ("cat.png", Png),
            ("a/b/icon.svg", Svg),
            ("photo.heic", Heic),
            ("photo.heif", Heic),
            ("scan.tiff", Tiff),
            ("scan.tif", Tiff),
            ("anim.webp", WebP),
        ] {
            assert!(check_upload_media(name, media).is_ok(), "{name}");
        }
        // Mismatches are refused, including the active
        // content spellings the audit names.
        for (name, media) in [
            ("promo.html", Png),
            ("icon.svg", Png),
            ("photo.png", Jpeg),
            ("photo.jpg", Svg),
            ("page.xml", Gif),
            ("pic.bmp", Png),
        ] {
            assert!(check_upload_media(name, media).is_err(), "{name}");
        }
        // No extension, nothing to bind; unlisted media
        // kinds are unbound.
        assert!(check_upload_media("cat", Png).is_ok());
        assert!(check_upload_media("cat.", Png).is_ok());
        assert!(check_upload_media("odd.html", Other).is_ok());
    }

    #[tokio::test]
    async fn s3_store_round_trips_through_the_fake() {
        let fake = crate::s3store::test_support::start_fake().await;
        let store = UploadStore::s3(
            "uploads".to_string(),
            "orig/".to_string(),
            Some(fake.endpoint.clone()),
            "us-east-1".to_string(),
            None,
            1024,
            false,
        );
        assert_eq!(
            store.source_ref("heroes/cat.jpg"),
            "s3://uploads/orig/heroes/cat.jpg"
        );
        assert!(!store.exists("heroes/cat.jpg").await);
        store.put("heroes/cat.jpg", b"pixels").await.unwrap();
        assert!(store.exists("heroes/cat.jpg").await);
        // The object landed verbatim under bucket + prefix,
        // no cache-entry framing.
        let objects = fake.objects.lock().unwrap();
        assert_eq!(
            objects.get("orig/heroes/cat.jpg").map(Vec::as_slice),
            Some(b"pixels".as_slice())
        );
    }

    #[tokio::test]
    async fn s3_put_new_creates_exactly_once_and_exists_is_a_head() {
        let fake = crate::s3store::test_support::start_fake().await;
        let store = UploadStore::s3(
            "uploads".to_string(),
            String::new(),
            Some(fake.endpoint.clone()),
            "us-east-1".to_string(),
            None,
            1024,
            false,
        );
        assert!(!store.exists("heroes/cat.jpg").await);
        store.put_new("heroes/cat.jpg", b"first").await.unwrap();
        assert!(store.exists("heroes/cat.jpg").await);
        // The conditional create refuses the second writer
        // atomically; the winner's bytes are untouched.
        let err = store.put_new("heroes/cat.jpg", b"second").await;
        assert!(matches!(err, Err(UploadError::AlreadyExists)), "{err:?}");
        {
            let objects = fake.objects.lock().unwrap();
            assert_eq!(
                objects.get("heroes/cat.jpg").map(Vec::as_slice),
                Some(b"first".as_slice())
            );
        }
        // Existence was answered with HEADs and the writes
        // were conditional PUTs: no GET ever downloaded the
        // existing object (audits R2-21, R2-23).
        let requests = fake.requests.lock().unwrap();
        assert!(
            requests.iter().any(|r| r == "HEAD heroes/cat.jpg"),
            "{requests:?}"
        );
        assert!(
            requests
                .iter()
                .any(|r| r == "PUT heroes/cat.jpg if-none-match"),
            "{requests:?}"
        );
        assert!(
            !requests.iter().any(|r| r.starts_with("GET ")),
            "{requests:?}"
        );
    }
}
