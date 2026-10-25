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
/// accepted. The handler maps these: a backend that is
/// unreachable is an upstream fault (502), a local IO
/// failure is a server fault (500).
#[derive(Debug)]
pub enum UploadError {
    /// The S3 backend refused or could not be reached.
    Backend(String),
    /// The local directory could not be written.
    Io(String),
}

impl std::fmt::Display for UploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
        }
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
            Backend::Local(_) => format!("file:///{name}"),
            Backend::S3 { bucket, prefix, .. } => format!("s3://{bucket}/{prefix}{name}"),
        }
    }

    /// Whether an object already exists under `name`.
    /// Backend errors read as "does not exist" for S3
    /// (its store contract degrades failures to misses);
    /// the write that follows surfaces a real failure.
    pub async fn exists(&self, name: &str) -> bool {
        match &self.backend {
            Backend::Local(dir) => dir.join(name).is_file(),
            Backend::S3 { store, .. } => store.get_raw(name).await.is_some(),
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
                tokio::task::spawn_blocking(move || put_local(&dir, &name, &bytes))
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

fn put_local(dir: &std::path::Path, name: &str, bytes: &[u8]) -> Result<(), UploadError> {
    let target = dir.join(name);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| UploadError::Io(format!("cannot create the upload directory: {e}")))?;
    }
    let tmp = target.with_file_name(tmp_name());
    std::fs::write(&tmp, bytes)
        .map_err(|e| UploadError::Io(format!("cannot write the upload: {e}")))?;
    if std::fs::rename(&tmp, &target).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return Err(UploadError::Io(
            "cannot move the upload into place".to_string(),
        ));
    }
    Ok(())
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
}
