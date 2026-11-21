use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

pub const SCHEMA: &str = "greppy.web-runtime.v1";

/// Immutable content identity (`sha256:<64 lowercase hex>`) for the executable image.
///
/// On Unix, canonical file metadata (`dev`, `ino`, `size`, `mtime`, `ctime`)
/// keys a private per-user digest cache under
/// `$XDG_CACHE_HOME/greppy/web-image-digest` (or
/// `$HOME/.cache/greppy/web-image-digest`). Entries are populated only from a
/// full streaming SHA-256 hash of the file, and any cache error or unsafe
/// directory/entry metadata falls back to hashing. The identity exposed over
/// the protocol is always the content digest, so reinstalling identical signed
/// bytes does not create a different runtime generation.
#[cfg(unix)]
pub fn runtime_image_id(path: &Path) -> io::Result<String> {
    let cache_dir = image_digest_cache_dir();
    runtime_image_id_with_cache_dir(path, cache_dir.as_deref())
}

#[cfg(unix)]
fn image_digest_cache_dir_from(
    xdg_cache_home: Option<&std::ffi::OsStr>,
    home: Option<&std::ffi::OsStr>,
) -> Option<std::path::PathBuf> {
    if let Some(xdg) = xdg_cache_home {
        let xdg = Path::new(xdg);
        if xdg.is_absolute() {
            return Some(xdg.join("greppy").join("web-image-digest"));
        }
    }
    let home = Path::new(home?);
    if home.is_absolute() {
        Some(home.join(".cache").join("greppy").join("web-image-digest"))
    } else {
        None
    }
}

#[cfg(unix)]
fn image_digest_cache_dir() -> Option<std::path::PathBuf> {
    let xdg = std::env::var_os("XDG_CACHE_HOME");
    let home = std::env::var_os("HOME");
    image_digest_cache_dir_from(xdg.as_deref(), home.as_deref())
}

#[cfg(unix)]
fn image_digest_cache_name(
    device: u64,
    inode: u64,
    size: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
) -> String {
    // The cache must use the same metadata identity as supervisor's
    // identity_mismatch. Same-size writes can restore mtime but still advance
    // ctime. Old keys without ctime deliberately miss; never migrate their
    // unbound digest.
    format!(
        "greppy-web-image-{device}-{inode}-{size}-{}-{}-{}-{}.sha256",
        mtime.0, mtime.1, ctime.0, ctime.1
    )
}

#[cfg(unix)]
fn image_digest_cache_name_from_meta(meta: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt;
    image_digest_cache_name(
        meta.dev(),
        meta.ino(),
        meta.size(),
        (meta.mtime(), meta.mtime_nsec()),
        (meta.ctime(), meta.ctime_nsec()),
    )
}

#[cfg(unix)]
fn ensure_private_cache_dir(dir: &Path) -> bool {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    let meta = match std::fs::symlink_metadata(dir) {
        Ok(meta) => meta,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let Some(parent) = dir.parent() else {
                return false;
            };
            if std::fs::create_dir_all(parent).is_err() {
                return false;
            }
            let mut builder = std::fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(dir) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
                Err(_) => return false,
            }
            match std::fs::symlink_metadata(dir) {
                Ok(meta) => meta,
                Err(_) => return false,
            }
        }
        Err(_) => return false,
    };
    let uid = unsafe { libc::geteuid() };
    let file_type = meta.file_type();
    file_type.is_dir() && !file_type.is_symlink() && meta.uid() == uid && (meta.mode() & 0o077) == 0
}

#[cfg(unix)]
fn read_image_digest_cache(cache_dir: &Path, key: &str) -> Option<String> {
    use std::io::Read;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let path = cache_dir.join(key);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .ok()?;
    let meta = file.metadata().ok()?;
    let uid = unsafe { libc::geteuid() };
    if !meta.file_type().is_file()
        || meta.uid() != uid
        || (meta.mode() & 0o077) != 0
        || meta.len() != 64
    {
        return None;
    }
    let mut buf = Vec::with_capacity(65);
    file.take(65).read_to_end(&mut buf).ok()?;
    if buf.len() != 64 || !buf.iter().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let digest = String::from_utf8(buf).ok()?.to_ascii_lowercase();
    Some(digest)
}

#[cfg(unix)]
fn write_image_digest_cache(cache_dir: &Path, key: &str, digest: &str) {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let tmp_path = cache_dir.join(format!(".{key}.tmp.{}", random_token()));
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&tmp_path)
    {
        Ok(file) => file,
        Err(_) => return,
    };
    if file.write_all(digest.as_bytes()).is_err() {
        drop(file);
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }
    drop(file);
    if std::fs::rename(&tmp_path, cache_dir.join(key)).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
}

#[cfg(unix)]
fn runtime_image_id_with_cache_dir(path: &Path, cache_dir: Option<&Path>) -> io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let cache_key = std::fs::canonicalize(path)
        .and_then(|canonical| std::fs::metadata(&canonical))
        .ok()
        .map(|meta| image_digest_cache_name_from_meta(&meta));
    let valid_cache_dir = match (cache_dir, cache_key.as_deref()) {
        (Some(dir), Some(_)) if ensure_private_cache_dir(dir) => Some(dir),
        _ => None,
    };
    if let (Some(dir), Some(key)) = (valid_cache_dir, cache_key.as_deref()) {
        if let Some(digest) = read_image_digest_cache(dir, key) {
            return Ok(format!("sha256:{digest}"));
        }
    }

    // Runtime images can be hundreds of MiB. Hash the same complete byte stream
    // without materializing a second image-sized anonymous allocation.
    let mut image = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = match image.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let digest = format!("{:x}", hash.finalize());
    if let (Some(dir), Some(key)) = (valid_cache_dir, cache_key.as_deref()) {
        if let Ok(after_meta) = image.metadata() {
            if image_digest_cache_name_from_meta(&after_meta) == key {
                write_image_digest_cache(dir, key, &digest);
            }
        }
    }
    Ok(format!("sha256:{digest}"))
}

#[cfg(not(unix))]
pub fn runtime_image_id(path: &Path) -> io::Result<String> {
    use std::time::UNIX_EPOCH;

    let metadata = std::fs::metadata(path)?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(format!("portable:{:x}:{modified:x}", metadata.len()))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Request {
    pub schema: String,
    pub request_id: String,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub deadline_ms: u64,
    pub operation: String,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub capability: String,
}

impl Request {
    pub fn new(
        run_id: impl Into<String>,
        operation: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            request_id: new_request_id(),
            run_id: run_id.into(),
            session_id: None,
            deadline_ms: 30_000,
            operation: operation.into(),
            payload,
            capability: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub schema: String,
    pub request_id: String,
    pub operation: String,
    pub status: String,
    pub result: Option<serde_json::Value>,
    pub artifacts: Vec<serde_json::Value>,
    pub metrics: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handshake: Option<Handshake>,
}

impl Response {
    pub fn ok(request: &Request, result: serde_json::Value) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            request_id: request.request_id.clone(),
            operation: request.operation.clone(),
            status: "ok".to_owned(),
            result: Some(result),
            artifacts: Vec::new(),
            metrics: Metrics::default(),
            error: None,
            handshake: None,
        }
    }

    pub fn error(request: &Request, error: ErrorObject) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            request_id: request.request_id.clone(),
            operation: request.operation.clone(),
            status: "error".to_owned(),
            result: None,
            artifacts: Vec::new(),
            metrics: Metrics::default(),
            error: Some(error),
            handshake: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorObject {
    pub code: String,
    pub message: Box<str>,
    pub operation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub retryable: bool,
    pub next_action: String,
    pub exit_code: i32,
}

impl ErrorObject {
    pub fn new(
        code: &str,
        message: impl Into<String>,
        operation_id: impl Into<String>,
        exit_code: i32,
        next_action: impl Into<String>,
    ) -> Self {
        Self {
            code: code.to_owned(),
            message: message.into().into_boxed_str(),
            operation_id: operation_id.into(),
            session_id: None,
            retryable: false,
            next_action: next_action.into(),
            exit_code,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Metrics {
    pub wall_ms: u64,
    pub controller_cpu_ms: u64,
    pub content_cpu_ms: u64,
    pub peak_rss_bytes: u64,
    pub network_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Handshake {
    pub protocol_version: String,
    pub runtime_build_id: String,
    pub playwright_compatibility_version: String,
    pub servo_revision: String,
    pub v8_revision: String,
    pub platform: String,
    pub architecture: String,
    pub supported_capabilities: Vec<String>,
    pub compatibility_coverage_level: String,
    pub max_message_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl Handshake {
    pub fn runtime_facts() -> Self {
        Self {
            protocol_version: SCHEMA.to_owned(),
            runtime_build_id: "web-runtime-0.1.0".to_owned(),
            playwright_compatibility_version: "1.62.1".to_owned(),
            servo_revision: "77fccacc1f1fdce10498d50173aafaa09d02879e".to_owned(),
            v8_revision: "deno_core-0.410.0".to_owned(),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            supported_capabilities: vec![
                "chromium.launch".into(),
                "session".into(),
                "web.run".into(),
                "web.observe".into(),
                "web.workflow.v1".into(),
                "web.inspect".into(),
                "web.screenshot".into(),
                "web.read".into(),
                "web.search".into(),
                "web.research".into(),
                "web.artifacts".into(),
                "page.route".into(),
                "page.frames".into(),
                "page.setInputFiles".into(),
            ],
            compatibility_coverage_level: "unverified".to_owned(),
            max_message_bytes: crate::MAX_FRAME_BYTES as u64,
            max_artifact_bytes: crate::MAX_FRAME_BYTES as u64,
        }
    }
}

pub fn new_request_id() -> String {
    format!("wrq_{}", random_token())
}

pub fn new_session_id() -> String {
    format!("wrs_{}", random_token())
}

fn random_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "{:x}{:x}{:x}",
        std::process::id(),
        nanos,
        SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn runtime_image_id_tracks_content_not_file_instance() {
        let root = std::env::temp_dir().join(format!(
            "greppy-runtime-image-id-{}-{}",
            std::process::id(),
            random_token()
        ));
        std::fs::create_dir_all(&root).unwrap();
        // Exercise the digest cache too, but never in the developer's real cache.
        let cache = root.join("cache");
        let runtime_image_id = |path: &Path| runtime_image_id_with_cache_dir(path, Some(&cache));
        let executable = root.join("web-runtime");
        let replacement = root.join("web-runtime.new");
        std::fs::write(&executable, b"old image").unwrap();
        let old = runtime_image_id(&executable).unwrap();
        std::fs::write(&replacement, b"new image").unwrap();
        std::fs::rename(&replacement, &executable).unwrap();
        std::fs::write(
            root.join("SHA256SUMS"),
            format!("{}  web-runtime\n", "0".repeat(64)),
        )
        .unwrap();
        let new = runtime_image_id(&executable).unwrap();
        assert_ne!(old, new);
        assert_ne!(new, format!("sha256:{}", "0".repeat(64)));

        let copy = root.join("copy");
        std::fs::copy(&executable, &copy).unwrap();
        assert_eq!(new, runtime_image_id(&copy).unwrap());

        // The identity must cover every byte, including partial and exact
        // streaming-buffer boundaries; retain the original whole-input oracle.
        use sha2::{Digest, Sha256};
        for length in [0, 1, 65_535, 65_536, 65_537, 131_089] {
            let bytes: Vec<u8> = (0..length).map(|index| (index % 251) as u8).collect();
            std::fs::write(&executable, &bytes).unwrap();
            assert_eq!(
                runtime_image_id(&executable).unwrap(),
                format!("sha256:{:x}", Sha256::digest(&bytes)),
                "complete content identity at {length} bytes",
            );
        }
        assert_eq!(
            runtime_image_id(&root.join("absent")).unwrap_err().kind(),
            io::ErrorKind::NotFound,
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn image_digest_cache_key_binds_every_metadata_identity_field() {
        let key = image_digest_cache_name(1, 2, 3, (4, 5), (6, 7));
        assert_eq!(key, image_digest_cache_name(1, 2, 3, (4, 5), (6, 7)));
        for changed in [
            image_digest_cache_name(9, 2, 3, (4, 5), (6, 7)),
            image_digest_cache_name(1, 9, 3, (4, 5), (6, 7)),
            image_digest_cache_name(1, 2, 9, (4, 5), (6, 7)),
            image_digest_cache_name(1, 2, 3, (9, 5), (6, 7)),
            image_digest_cache_name(1, 2, 3, (4, 9), (6, 7)),
            image_digest_cache_name(1, 2, 3, (4, 5), (9, 7)),
            image_digest_cache_name(1, 2, 3, (4, 5), (6, 9)),
        ] {
            assert_ne!(key, changed);
        }
        assert_ne!(key, "greppy-web-image-1-2-3-4-5.sha256");
    }

    #[cfg(unix)]
    #[test]
    fn runtime_image_id_caches_by_metadata_and_rejects_unsafe_entries() {
        use sha2::{Digest, Sha256};
        use std::ffi::OsStr;
        use std::io::Write;
        use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
        use std::time::Duration;

        let root = std::env::temp_dir().join(format!(
            "greppy-runtime-image-cache-{}-{}",
            std::process::id(),
            random_token()
        ));
        let xdg = root.join("xdg");
        std::fs::create_dir_all(&xdg).unwrap();

        let cache_dir = image_digest_cache_dir_from(Some(xdg.as_os_str()), None).unwrap();
        assert_eq!(cache_dir, xdg.join("greppy").join("web-image-digest"));
        assert_eq!(
            image_digest_cache_dir_from(Some(OsStr::new("relative")), Some(root.as_os_str())),
            Some(root.join(".cache").join("greppy").join("web-image-digest"))
        );

        let executable = root.join("web-runtime");
        std::fs::write(&executable, b"original-bytes").unwrap();

        // (a) Second call returns the same id and creates exactly one cache entry.
        let first = runtime_image_id_with_cache_dir(&executable, Some(&cache_dir)).unwrap();
        let second = runtime_image_id_with_cache_dir(&executable, Some(&cache_dir)).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            std::fs::symlink_metadata(&cache_dir).unwrap().mode() & 0o777,
            0o700
        );
        let entries: Vec<_> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            std::fs::symlink_metadata(&entries[0]).unwrap().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::read_to_string(&entries[0]).unwrap(),
            first.strip_prefix("sha256:").unwrap()
        );

        // (b) In-place same-length rewrite advances mtime/ctime and misses the cache.
        let before_ino = std::fs::metadata(&executable).unwrap().ino();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::OpenOptions::new()
            .write(true)
            .open(&executable)
            .unwrap()
            .write_all(b"XXXXXXXXXXXXXX")
            .unwrap();
        let after_meta = std::fs::metadata(&executable).unwrap();
        assert_eq!(after_meta.ino(), before_ino);
        assert_eq!(after_meta.len(), b"original-bytes".len() as u64);
        let rewritten = runtime_image_id_with_cache_dir(&executable, Some(&cache_dir)).unwrap();
        assert_ne!(first, rewritten);
        assert_eq!(
            rewritten,
            format!("sha256:{:x}", Sha256::digest(b"XXXXXXXXXXXXXX"))
        );

        // (c) Cache entries with unsafe permissions (0644) or symlinks are ignored.
        let canonical = std::fs::canonicalize(&executable).unwrap();
        let key = image_digest_cache_name_from_meta(&std::fs::metadata(&canonical).unwrap());
        let entry_path = cache_dir.join(&key);

        std::fs::write(&entry_path, "0".repeat(64)).unwrap();
        std::fs::set_permissions(&entry_path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let after_bad_perm =
            runtime_image_id_with_cache_dir(&executable, Some(&cache_dir)).unwrap();
        assert_eq!(after_bad_perm, rewritten);

        let poison = root.join("poison");
        std::fs::write(&poison, "f".repeat(64)).unwrap();
        std::fs::set_permissions(&poison, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::remove_file(&entry_path).unwrap();
        symlink(&poison, &entry_path).unwrap();
        let after_symlink = runtime_image_id_with_cache_dir(&executable, Some(&cache_dir)).unwrap();
        assert_eq!(after_symlink, rewritten);
        assert_eq!(std::fs::read_to_string(&poison).unwrap(), "f".repeat(64));

        // Unsafe cache directory (group/other readable) falls back to hashing without writing.
        let bad_dir = root.join("world-readable-cache");
        std::fs::create_dir_all(&bad_dir).unwrap();
        std::fs::set_permissions(&bad_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(
            runtime_image_id_with_cache_dir(&executable, Some(&bad_dir)).unwrap(),
            rewritten
        );
        assert_eq!(std::fs::read_dir(&bad_dir).unwrap().count(), 0);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn request_ids_use_required_prefix() {
        assert!(new_request_id().starts_with("wrq_"));
        assert!(new_session_id().starts_with("wrs_"));
    }
}
