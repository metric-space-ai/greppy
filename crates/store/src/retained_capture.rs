//! Private, quota-bounded disk backing for bash-smart packs. Pack rows contain
//! only metadata; continuation rows share the same immutable pair of files.
use crate::store::Store;
use crate::store_error::{Error, Result};
use greppy_core::cache::{acquire_named_lock_in, LockMode};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

const RETAINED_QUOTA_BYTES: u64 = 512 * 1024 * 1024;
const RETAINED_MAX_FILES: usize = 2048;
const STREAM_CAP_BYTES: u64 = 64 * 1024 * 1024 + 256;

/// Byte and file ceilings for one retained-capture namespace.
///
/// Production uses the constants above. Tests inject a tiny ceiling through
/// `RetainedLimits::new` so eviction can be proven without a 512 MiB fixture.
#[derive(Clone, Copy)]
struct RetainedLimits {
    bytes: u64,
    max_files: usize,
}

impl RetainedLimits {
    fn production() -> Self {
        Self {
            bytes: RETAINED_QUOTA_BYTES,
            max_files: RETAINED_MAX_FILES,
        }
    }

    /// Test-only constructor. Not used by the production retention path.
    #[cfg(test)]
    fn new(bytes: u64, max_files: usize) -> Self {
        Self { bytes, max_files }
    }
}

impl Store {
    fn retained_capture_dir(&self) -> Result<PathBuf> {
        let path: String = self
            .conn()
            .query_row("PRAGMA database_list", [], |row| row.get(2))?;
        if path.is_empty() {
            return Err(Error::Store(
                "retained capture requires a file-backed store".into(),
            ));
        }
        let key = format!("{:x}", Sha256::digest(path.as_bytes()));
        // A mounted disposable macOS volume keeps large captures off the system
        // disk; without one (every ordinary Mac) they live next to the store.
        Ok(match greppy_core::cache::macos_disposable_volume() {
            Some(volume) => volume.join("dev-artifacts/greppy/bash-smart-retained"),
            None => Path::new(&path).with_file_name("bash-smart-retained"),
        }
        .join(key))
    }

    pub fn retain_bash_smart_capture(
        &self,
        payload: &serde_json::Value,
        stdout: &[u8],
        stderr: &[u8],
    ) -> Result<serde_json::Value> {
        if stdout.len() as u64 > STREAM_CAP_BYTES || stderr.len() as u64 > STREAM_CAP_BYTES {
            return Err(Error::Store("retained capture exceeds stream cap".into()));
        }
        // retained_capture_dir only selects /Volumes/tmp when it is a mounted
        // volume of its own; otherwise captures live next to the store.
        let dir = self.retained_capture_dir()?;
        let root = dir
            .parent()
            .ok_or_else(|| Error::Store("retained namespace has no owner".into()))?;
        greppy_core::workspace::ensure_store_dir(root)
            .map_err(|e| Error::Store(format!("create retained capture namespace: {e}")))?;
        let _lease = acquire_named_lock_in(root, "capture", LockMode::Exclusive, true)
            .map_err(|e| Error::Store(format!("lease retained capture: {e}")))?
            .ok_or_else(|| Error::Store("retained capture namespace busy".into()))?;
        prune_namespace(root, now())?;
        greppy_core::workspace::ensure_store_dir(&dir)
            .map_err(|e| Error::Store(format!("create retained capture workspace: {e}")))?;
        let until = payload
            .get("retained_until")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Store("retained capture deadline missing".into()))?;
        if until <= now() {
            return Err(Error::Store("retained capture deadline elapsed".into()));
        }
        let hash = payload
            .get("content_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|value| valid_hash(value))
            .ok_or_else(|| Error::Store("retained capture content hash invalid".into()))?;
        let paths = [
            dir.join(format!("{until}-{hash}.stdout")),
            dir.join(format!("{until}-{hash}.stderr")),
        ];
        check_quota(root, &paths, [stdout, stderr], active_limits())?;
        for name in ["stdout", "stderr"] {
            if !payload.get(name).is_some_and(serde_json::Value::is_object) {
                return Err(Error::Store("retained stream metadata missing".into()));
            }
        }
        let new_paths: Vec<_> = paths
            .iter()
            .filter(|path| !path.exists())
            .cloned()
            .collect();
        let mut retained = payload.clone();
        for ((name, bytes), path) in [("stdout", stdout), ("stderr", stderr)]
            .into_iter()
            .zip(paths)
        {
            if let Err(error) = publish_artifact(&path, bytes) {
                for created in &new_paths {
                    let _ = std::fs::remove_file(created);
                }
                return Err(error);
            }
            let stream = retained
                .get_mut(name)
                .and_then(serde_json::Value::as_object_mut)
                .ok_or_else(|| Error::Store("retained stream metadata missing".into()))?;
            if let Some(original) = stream.get("path").cloned() {
                stream.insert("capture_path".into(), original);
            }
            stream.insert("path".into(), serde_json::json!(path));
        }
        #[cfg(unix)]
        if let Err(error) = std::fs::File::open(&dir).and_then(|directory| directory.sync_all()) {
            for created in &new_paths {
                let _ = std::fs::remove_file(created);
            }
            return Err(Error::Store(format!(
                "persist retained capture directory: {error}"
            )));
        }
        retained["retained_artifact_dir"] = serde_json::json!(root);
        Ok(retained)
    }

    pub(crate) fn prune_retained_captures_at(&self, cutoff: u64) -> Result<()> {
        let Ok(dir) = self.retained_capture_dir() else {
            return Ok(());
        };
        let Some(root) = dir.parent() else {
            return Ok(());
        };
        if !root.exists() {
            return Ok(());
        }
        // Never wait behind an active reader or writer just to collect garbage.
        try_prune_namespace(root, cutoff)
    }
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn artifact_deadline(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let (until, rest) = name.split_once('-')?;
    let (hash, stream) = rest.split_once('.')?;
    (valid_hash(hash) && matches!(stream, "stdout" | "stderr")).then_some(())?;
    until.parse().ok()
}

fn owned_workspaces(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(root)
        .map_err(|e| Error::Store(format!("scan retained capture owner: {e}")))?
    {
        let entry =
            entry.map_err(|e| Error::Store(format!("inspect retained capture owner: {e}")))?;
        if entry.file_name().to_str().is_some_and(valid_hash) {
            if !entry
                .file_type()
                .map_err(|e| Error::Store(format!("inspect retained workspace: {e}")))?
                .is_dir()
            {
                return Err(Error::Store(
                    "retained workspace is not a regular directory".into(),
                ));
            }
            dirs.push(entry.path());
        }
    }
    Ok(dirs)
}

fn active_limits() -> RetainedLimits {
    // Test seam only: release builds always use the production limits, so a
    // stray CI variable can never shrink the machine-wide namespace and evict
    // users' captures. Both knobs are required.
    #[cfg(debug_assertions)]
    if let (Ok(bytes), Ok(max_files)) = (
        std::env::var("GREPPY_TEST_RETAINED_QUOTA_BYTES"),
        std::env::var("GREPPY_TEST_RETAINED_MAX_FILES"),
    ) {
        if let (Ok(bytes), Ok(max_files)) = (bytes.parse::<u64>(), max_files.parse::<usize>()) {
            return RetainedLimits { bytes, max_files };
        }
    }
    RetainedLimits::production()
}

fn check_quota(
    root: &Path,
    paths: &[PathBuf; 2],
    streams: [&[u8]; 2],
    limits: RetainedLimits,
) -> Result<()> {
    let extra: u64 = paths
        .iter()
        .zip(streams)
        .filter(|(path, _)| !path.exists())
        .map(|(_, bytes)| bytes.len() as u64)
        .sum();
    let new_files = paths.iter().filter(|path| !path.exists()).count();
    // A capture that cannot fit even in an empty namespace is refused before
    // any eviction. Deleting older captures would not make room.
    if extra > limits.bytes || new_files > limits.max_files {
        return Err(Error::Store(
            "retained capture exceeds quota by itself".into(),
        ));
    }
    loop {
        let (usage, file_count) = namespace_usage(root)?;
        if usage.saturating_add(extra) <= limits.bytes && file_count + new_files <= limits.max_files
        {
            return Ok(());
        }
        // Never refuse while some other capture can be deleted. Oldest deadline
        // goes first so a full namespace keeps the newest expand ids.
        if !evict_oldest_capture(root, paths)? {
            return Err(Error::Store(
                "retained capture quota exhausted; command output remains in its raw capture paths"
                    .into(),
            ));
        }
    }
}

struct ArtifactFile {
    path: PathBuf,
    deadline: u64,
    group: String,
}

fn list_artifacts(root: &Path) -> Result<Vec<ArtifactFile>> {
    let mut artifacts = Vec::new();
    for dir in owned_workspaces(root)? {
        for entry in std::fs::read_dir(&dir)
            .map_err(|e| Error::Store(format!("scan retained capture quota: {e}")))?
        {
            let entry =
                entry.map_err(|e| Error::Store(format!("inspect retained artifact quota: {e}")))?;
            let path = entry.path();
            let Some(deadline) = artifact_deadline(&path) else {
                continue;
            };
            let Some(name) = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned)
            else {
                continue;
            };
            let Some((group, _)) = name.rsplit_once('.') else {
                continue;
            };
            artifacts.push(ArtifactFile {
                path,
                deadline,
                group: group.to_string(),
            });
        }
    }
    Ok(artifacts)
}

/// Delete every stream file of the oldest-deadline capture, except the paths
/// this retention is about to publish. Returns false when nothing remains to
/// evict.
fn evict_oldest_capture(root: &Path, protect: &[PathBuf; 2]) -> Result<bool> {
    let mut artifacts = list_artifacts(root)?;
    artifacts.retain(|artifact| !protect.iter().any(|path| path == &artifact.path));
    artifacts.sort_by(|left, right| {
        left.deadline
            .cmp(&right.deadline)
            .then_with(|| left.group.cmp(&right.group))
            .then_with(|| left.path.cmp(&right.path))
    });
    // Oldest capture group first. A group whose files cannot be removed (for
    // example owned by another user in a shared root) is skipped, so one stuck
    // artifact never disables retention for the whole namespace.
    let mut index = 0;
    while index < artifacts.len() {
        let deadline = artifacts[index].deadline;
        let group = artifacts[index].group.clone();
        let mut removed = false;
        while index < artifacts.len()
            && artifacts[index].deadline == deadline
            && artifacts[index].group == group
        {
            match std::fs::remove_file(&artifacts[index].path) {
                Ok(()) => removed = true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => removed = true,
                Err(_) => {}
            }
            index += 1;
        }
        if removed {
            return Ok(true);
        }
    }
    Ok(false)
}

fn namespace_usage(root: &Path) -> Result<(u64, usize)> {
    let mut bytes = 0u64;
    let mut files = 0usize;
    for dir in owned_workspaces(root)? {
        for entry in std::fs::read_dir(dir)
            .map_err(|e| Error::Store(format!("scan retained capture quota: {e}")))?
        {
            let entry =
                entry.map_err(|e| Error::Store(format!("inspect retained artifact quota: {e}")))?;
            if artifact_deadline(&entry.path()).is_some() {
                let meta = std::fs::symlink_metadata(entry.path())
                    .map_err(|e| Error::Store(format!("inspect retained artifact: {e}")))?;
                if !meta.is_file() {
                    return Err(Error::Store(
                        "retained artifact is not a regular file".into(),
                    ));
                }
                bytes = bytes.saturating_add(meta.len());
                files += 1;
            }
        }
    }
    Ok((bytes, files))
}

fn try_prune_namespace(root: &Path, cutoff: u64) -> Result<()> {
    let lease = acquire_named_lock_in(root, "capture", LockMode::Exclusive, true)
        .map_err(|e| Error::Store(format!("lease retained capture cleanup: {e}")))?;
    if let Some(_lease) = lease {
        prune_namespace(root, cutoff)?;
    }
    Ok(())
}

fn prune_namespace(root: &Path, cutoff: u64) -> Result<()> {
    for dir in owned_workspaces(root)? {
        prune_artifacts(&dir, cutoff)?;
        // Handles no longer own bytes after their TTL plus a short read grace.
        // Empty abandoned workspace namespaces are also bounded and reclaimed.
        match std::fs::remove_dir(&dir) {
            Ok(()) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::DirectoryNotEmpty | std::io::ErrorKind::NotFound
                ) => {}
            Err(error) => {
                return Err(Error::Store(format!(
                    "remove empty retained workspace: {error}"
                )))
            }
        }
    }
    Ok(())
}

fn prune_artifacts(dir: &Path, cutoff: u64) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).map_err(|e| Error::Store(format!("scan retained captures: {e}")))?
    {
        let entry = entry.map_err(|e| Error::Store(format!("inspect retained capture: {e}")))?;
        if artifact_deadline(&entry.path()).is_some_and(|until| until <= cutoff) {
            // Remove only this namespace's recognized files, never arbitrary
            // metadata paths supplied by a pack or a recursively scanned tree.
            std::fs::remove_file(entry.path())
                .map_err(|e| Error::Store(format!("remove expired retained capture: {e}")))?;
        }
    }
    Ok(())
}

fn publish_artifact(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(mut file) => {
            if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
                drop(file);
                let _ = std::fs::remove_file(path);
                return Err(Error::Store(format!("persist retained capture: {error}")));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let meta = std::fs::symlink_metadata(path)
                .map_err(|e| Error::Store(format!("inspect existing retained capture: {e}")))?;
            if !meta.is_file() || meta.len() != bytes.len() as u64 {
                return Err(Error::Store(
                    "existing retained capture is not verified regular bytes".into(),
                ));
            }
            // Streaming verification does not clone the retained capture.
            use std::io::Read;
            let mut read_options = std::fs::OpenOptions::new();
            read_options.read(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                read_options.custom_flags(libc::O_NOFOLLOW);
            }
            let mut file = read_options
                .open(path)
                .map_err(|e| Error::Store(format!("open retained capture: {e}")))?;
            let mut seen = 0u64;
            let mut actual = Sha256::new();
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let count = file
                    .read(&mut buffer)
                    .map_err(|e| Error::Store(format!("verify retained capture: {e}")))?;
                if count == 0 {
                    break;
                }
                seen = seen.saturating_add(count as u64);
                if seen > bytes.len() as u64 {
                    return Err(Error::Store(
                        "existing retained capture grew beyond verified byte bound".into(),
                    ));
                }
                actual.update(&buffer[..count]);
            }
            if actual.finalize() != Sha256::digest(bytes) {
                return Err(Error::Store("existing retained capture hash drift".into()));
            }
        }
        Err(error) => return Err(Error::Store(format!("create retained capture: {error}"))),
    }
    Ok(())
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_pruning_respects_readers_live_artifacts_and_abandoned_workspaces() {
        let root = tempfile::tempdir().unwrap();
        let expired_dir = root.path().join("a".repeat(64));
        let live_dir = root.path().join("b".repeat(64));
        std::fs::create_dir(&expired_dir).unwrap();
        std::fs::create_dir(&live_dir).unwrap();
        let expired = expired_dir.join(format!("10-{}.stdout", "c".repeat(64)));
        let live = live_dir.join(format!("100-{}.stdout", "d".repeat(64)));
        publish_artifact(&expired, b"expired").unwrap();
        publish_artifact(&live, b"live").unwrap();
        let first_reader = acquire_named_lock_in(root.path(), "capture", LockMode::Shared, false)
            .unwrap()
            .unwrap();
        let second_reader = acquire_named_lock_in(root.path(), "capture", LockMode::Shared, true)
            .unwrap()
            .expect("concurrent readers");
        try_prune_namespace(root.path(), 50).unwrap();
        assert!(
            expired.exists(),
            "active readers keep files through collection"
        );
        drop(first_reader);
        try_prune_namespace(root.path(), 50).unwrap();
        assert!(expired.exists());
        drop(second_reader);
        try_prune_namespace(root.path(), 50).unwrap();
        assert!(
            !expired_dir.exists(),
            "expired abandoned namespace reclaimed"
        );
        assert_eq!(std::fs::read(&live).unwrap(), b"live");
        assert_eq!(namespace_usage(root.path()).unwrap(), (4, 1));
    }

    fn artifact_hex(dir: &Path, deadline: u64, hex: char, stream: &str, bytes: &[u8]) -> PathBuf {
        let hash = hex.to_string().repeat(64);
        let path = dir.join(format!("{deadline}-{hash}.{stream}"));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn quota_evicts_oldest_deadline_across_workspaces_until_the_capture_fits() {
        let root = tempfile::tempdir().unwrap();
        let old_dir = root.path().join("a".repeat(64));
        let newer_dir = root.path().join("b".repeat(64));
        let new_dir = root.path().join("c".repeat(64));
        std::fs::create_dir(&old_dir).unwrap();
        std::fs::create_dir(&newer_dir).unwrap();
        std::fs::create_dir(&new_dir).unwrap();
        let oldest_stdout = artifact_hex(&old_dir, 10, '1', "stdout", b"0123456789");
        let oldest_stderr = artifact_hex(&old_dir, 10, '1', "stderr", b"abcdefghij");
        let newer = artifact_hex(&newer_dir, 50, '2', "stdout", b"wxyz");
        let paths = [
            new_dir.join(format!("80-{}.stdout", "3".repeat(64))),
            new_dir.join(format!("80-{}.stderr", "3".repeat(64))),
        ];
        // 10+10+4 existing bytes. The new capture is 6 bytes / 2 files. A 12-byte
        // ceiling cannot hold the oldest capture plus the new one, but can hold
        // the newer 4-byte capture once the deadline-10 pair is gone.
        let limits = RetainedLimits::new(12, 4);
        check_quota(root.path(), &paths, [b"hello!", b""], limits).unwrap();
        assert!(!oldest_stdout.exists() && !oldest_stderr.exists());
        assert_eq!(std::fs::read(&newer).unwrap(), b"wxyz");
        assert!(!paths[0].exists() && !paths[1].exists());
        assert_eq!(namespace_usage(root.path()).unwrap(), (4, 1));
    }

    #[test]
    fn quota_evicts_for_file_count_and_stops_when_the_capture_fits() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("d".repeat(64));
        std::fs::create_dir(&dir).unwrap();
        let first = artifact_hex(&dir, 1, 'a', "stdout", b"a");
        let second = artifact_hex(&dir, 2, 'b', "stdout", b"b");
        let third = artifact_hex(&dir, 3, 'c', "stdout", b"c");
        let paths = [
            dir.join(format!("9-{}.stdout", "d".repeat(64))),
            dir.join(format!("9-{}.stderr", "d".repeat(64))),
        ];
        check_quota(
            root.path(),
            &paths,
            [b"x", b"y"],
            RetainedLimits::new(100, 3),
        )
        .unwrap();
        assert!(!first.exists() && !second.exists());
        assert_eq!(std::fs::read(&third).unwrap(), b"c");
        assert_eq!(namespace_usage(root.path()).unwrap(), (1, 1));
    }

    #[test]
    fn quota_refuses_a_capture_that_alone_exceeds_limits_without_evicting() {
        let root = tempfile::tempdir().unwrap();
        let old_dir = root.path().join("e".repeat(64));
        let new_dir = root.path().join("f".repeat(64));
        std::fs::create_dir(&old_dir).unwrap();
        std::fs::create_dir(&new_dir).unwrap();
        let kept = artifact_hex(&old_dir, 10, '9', "stdout", b"kept!");
        let paths = [
            new_dir.join(format!("20-{}.stdout", "8".repeat(64))),
            new_dir.join(format!("20-{}.stderr", "8".repeat(64))),
        ];
        let error = check_quota(
            root.path(),
            &paths,
            [b"0123456789", b"x"],
            RetainedLimits::new(8, 4),
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("exceeds quota by itself"),
            "{error}"
        );
        assert_eq!(std::fs::read(&kept).unwrap(), b"kept!");
        assert!(!paths[0].exists() && !paths[1].exists());

        let too_many_files = check_quota(
            root.path(),
            &paths,
            [b"x", b"y"],
            RetainedLimits::new(100, 1),
        )
        .unwrap_err();
        assert!(too_many_files
            .to_string()
            .contains("exceeds quota by itself"));
        assert_eq!(std::fs::read(&kept).unwrap(), b"kept!");
    }

    #[test]
    fn existing_artifact_tamper_and_symlink_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("capture");
        publish_artifact(&path, b"correct").unwrap();
        publish_artifact(&path, b"correct").unwrap();
        std::fs::write(&path, b"changed").unwrap();
        assert!(publish_artifact(&path, b"correct").is_err());
        #[cfg(unix)]
        {
            std::fs::remove_file(&path).unwrap();
            let target = root.path().join("target");
            std::fs::write(&target, b"correct").unwrap();
            std::os::unix::fs::symlink(&target, &path).unwrap();
            assert!(publish_artifact(&path, b"correct").is_err());
        }
    }
}
