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
        #[cfg(target_os = "macos")]
        {
            Ok(Path::new("/Volumes/tmp/dev-artifacts/greppy/bash-smart-retained").join(key))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Path::new(&path)
                .with_file_name("bash-smart-retained")
                .join(key))
        }
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
        let dir = self.retained_capture_dir()?;
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::fs::MetadataExt;
            let volume = std::fs::metadata("/Volumes/tmp").map_err(|e| {
                Error::Store(format!("retained capture tmp volume unavailable: {e}"))
            })?;
            let system = std::fs::metadata("/")
                .map_err(|e| Error::Store(format!("inspect system volume: {e}")))?;
            if !volume.is_dir() || volume.dev() == system.dev() {
                return Err(Error::Store(
                    "retained capture tmp volume is not mounted".into(),
                ));
            }
        }
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
        check_quota(root, &paths, [stdout, stderr])?;
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

fn check_quota(root: &Path, paths: &[PathBuf; 2], streams: [&[u8]; 2]) -> Result<()> {
    let (usage, file_count) = namespace_usage(root)?;
    let extra: u64 = paths
        .iter()
        .zip(streams)
        .filter(|(path, _)| !path.exists())
        .map(|(_, bytes)| bytes.len() as u64)
        .sum();
    let new_files = paths.iter().filter(|path| !path.exists()).count();
    if usage.saturating_add(extra) > RETAINED_QUOTA_BYTES
        || file_count + new_files > RETAINED_MAX_FILES
    {
        return Err(Error::Store(
            "retained capture quota exhausted; command output remains in its raw capture paths"
                .into(),
        ));
    }
    Ok(())
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

    #[test]
    fn quota_counts_all_workspaces_and_rejects_without_writing() {
        let root = tempfile::tempdir().unwrap();
        let old_dir = root.path().join("a".repeat(64));
        let new_dir = root.path().join("b".repeat(64));
        std::fs::create_dir(&old_dir).unwrap();
        std::fs::create_dir(&new_dir).unwrap();
        let old = old_dir.join(format!("100-{}.stdout", "c".repeat(64)));
        let file = std::fs::File::create(&old).unwrap();
        // Sparse size proves accounting without a substantial fixture allocation.
        file.set_len(RETAINED_QUOTA_BYTES).unwrap();
        let paths = [
            new_dir.join(format!("100-{}.stdout", "d".repeat(64))),
            new_dir.join(format!("100-{}.stderr", "d".repeat(64))),
        ];
        assert!(check_quota(root.path(), &paths, [b"x", b"y"]).is_err());
        assert!(!paths[0].exists() && !paths[1].exists());
        prune_namespace(root.path(), 100).unwrap();
        assert!(check_quota(root.path(), &paths, [b"x", b"y"]).is_ok());
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

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
