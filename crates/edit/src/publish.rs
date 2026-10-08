//! Publishing: atomic single-file writes with compare-and-swap, and the
//! pre-image journal for logical multi-file transactions.
//!
//! `atomic`: tmp file in the target directory + fsync + rename + directory
//! fsync; permissions preserved; symlinks and paths escaping the workspace
//! rejected. The live file hash is re-verified against the plan hash under
//! the same directory immediately before the rename — a concurrent change
//! after that window is indistinguishable from one after any other write and
//! is out of scope for single-file mode (journal mode takes a workspace
//! lock).

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::hash::sha256_hex;
use greppy_core::{Error, Result};

/// Reject paths that escape the workspace root or pass through symlinks.
pub fn require_inside_workspace(workspace_root: &Path, path: &Path) -> Result<PathBuf> {
    let root = workspace_root.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", workspace_root.display()),
        source,
    })?;
    // canonicalize the parent (the file itself may be replaced), then reattach
    let parent = path.parent().unwrap_or(Path::new("."));
    let parent = parent.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", parent.display()),
        source,
    })?;
    let file_name = path
        .file_name()
        .ok_or_else(|| Error::Invalid(format!("not a file path: {}", path.display())))?;
    let resolved = parent.join(file_name);
    if !resolved.starts_with(&root) {
        return Err(Error::Workspace(format!(
            "path {} escapes workspace {}",
            resolved.display(),
            root.display()
        )));
    }
    let meta = std::fs::symlink_metadata(&resolved).map_err(|source| Error::Io {
        context: format!("stat {}", resolved.display()),
        source,
    })?;
    if meta.file_type().is_symlink() {
        return Err(Error::Workspace(format!(
            "refusing to publish through symlink: {}",
            resolved.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.nlink() > 1 {
            return Err(Error::Workspace(format!(
                "refusing to publish through hardlinked file: {}",
                resolved.display()
            )));
        }
    }
    Ok(resolved)
}

/// Atomically replace `path` with `content`, iff the live file still hashes
/// to `expected_live_sha256`. Preserves permissions. Returns the new sha.
pub fn publish_atomic(
    workspace_root: &Path,
    path: &Path,
    content: &[u8],
    expected_live_sha256: &str,
) -> Result<String> {
    let resolved = require_inside_workspace(workspace_root, path)?;

    // compare-and-swap: re-read and re-hash immediately before the swap
    let live = std::fs::read(&resolved).map_err(|source| Error::Io {
        context: format!("read {}", resolved.display()),
        source,
    })?;
    if sha256_hex(&live) != expected_live_sha256 {
        return Err(Error::Workspace(format!(
            "stale plan: {} changed since planning; nothing was written",
            resolved.display()
        )));
    }

    let permissions = std::fs::metadata(&resolved)
        .map_err(|source| Error::Io {
            context: format!("stat {}", resolved.display()),
            source,
        })?
        .permissions();

    let dir = resolved.parent().unwrap_or(Path::new("."));
    let mut tmp = tempfile::Builder::new()
        .prefix(".greppy-edit.")
        .tempfile_in(dir)
        .map_err(|source| Error::Io {
            context: format!("tempfile in {}", dir.display()),
            source,
        })?;
    tmp.write_all(content).map_err(|source| Error::Io {
        context: "write tmp".into(),
        source,
    })?;
    tmp.as_file().sync_all().map_err(|source| Error::Io {
        context: "fsync tmp".into(),
        source,
    })?;
    std::fs::set_permissions(tmp.path(), permissions).map_err(|source| Error::Io {
        context: "preserve permissions".into(),
        source,
    })?;
    tmp.persist(&resolved).map_err(|e| Error::Io {
        context: format!("rename over {}", resolved.display()),
        source: e.error,
    })?;
    if let Ok(dir_handle) = std::fs::File::open(dir) {
        let _ = dir_handle.sync_all();
    }
    Ok(sha256_hex(content))
}

/// Resolve an absent target through its nearest existing ancestor without
/// writing directories. Missing paths still have to remain inside the root.
pub fn require_new_inside_workspace(workspace_root: &Path, path: &Path) -> Result<PathBuf> {
    let root = workspace_root.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", workspace_root.display()),
        source,
    })?;
    let mut ancestor = path
        .parent()
        .ok_or_else(|| Error::Invalid("not a file path".into()))?;
    let mut tail = Vec::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .ok_or_else(|| Error::Invalid("invalid creation parent".into()))?;
        tail.push(name.to_os_string());
        ancestor = ancestor
            .parent()
            .ok_or_else(|| Error::Invalid("invalid creation parent".into()))?;
    }
    let mut parent = ancestor.canonicalize().map_err(|source| Error::Io {
        context: format!("canonicalize {}", ancestor.display()),
        source,
    })?;
    for name in tail.into_iter().rev() {
        parent.push(name);
    }
    let name = path
        .file_name()
        .ok_or_else(|| Error::Invalid("not a file path".into()))?;
    let resolved = parent.join(name);
    if !resolved.starts_with(&root) {
        return Err(Error::Workspace(format!(
            "path {} escapes workspace {}",
            resolved.display(),
            root.display()
        )));
    }
    match std::fs::symlink_metadata(&resolved) {
        Ok(_) => Err(Error::Workspace(format!(
            "creation target {} already exists; nothing written",
            resolved.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(resolved),
        Err(source) => Err(Error::Io {
            context: format!("stat {}", resolved.display()),
            source,
        }),
    }
}

/// Publish a complete new file without replacing a racing creator. Temporary
/// bytes are private until persist_noclobber atomically links the final name.
pub fn publish_create_atomic(
    workspace_root: &Path,
    path: &Path,
    content: &[u8],
    permissions: Option<std::fs::Permissions>,
) -> Result<String> {
    let resolved = require_new_inside_workspace(workspace_root, path)?;
    let parent = resolved
        .parent()
        .ok_or_else(|| Error::Invalid("not a file path".into()))?;
    std::fs::create_dir_all(parent).map_err(|source| Error::Io {
        context: format!("create {}", parent.display()),
        source,
    })?;
    // Re-check after directory creation; an ancestor may have been replaced.
    let checked = require_new_inside_workspace(workspace_root, path)?;
    if checked != resolved {
        return Err(Error::Workspace(
            "creation parent changed since planning; nothing written".into(),
        ));
    }
    let mut tmp = tempfile::Builder::new()
        .prefix(".greppy-edit.")
        .tempfile_in(parent)
        .map_err(|source| Error::Io {
            context: "create temporary file".into(),
            source,
        })?;
    tmp.write_all(content).map_err(|source| Error::Io {
        context: "write tmp".into(),
        source,
    })?;
    if let Some(permissions) = permissions {
        std::fs::set_permissions(tmp.path(), permissions).map_err(|source| Error::Io {
            context: "restore permissions".into(),
            source,
        })?;
    }
    tmp.as_file().sync_all().map_err(|source| Error::Io {
        context: "fsync tmp".into(),
        source,
    })?;
    tmp.persist_noclobber(&resolved)
        .map_err(|error| Error::Io {
            context: format!("create {} without replacement", resolved.display()),
            source: error.error,
        })?;
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(sha256_hex(content))
}

/// Delete only the planned file bytes, with the same path/link guards and
/// compare-before-publication window as publish_atomic.
pub fn publish_delete_atomic(
    workspace_root: &Path,
    path: &Path,
    expected_live_sha256: &str,
) -> Result<()> {
    let resolved = require_inside_workspace(workspace_root, path)?;
    let live = std::fs::read(&resolved).map_err(|source| Error::Io {
        context: format!("read {}", resolved.display()),
        source,
    })?;
    if sha256_hex(&live) != expected_live_sha256 {
        return Err(Error::Workspace(format!(
            "stale plan: {} changed since planning; nothing was written",
            resolved.display()
        )));
    }
    std::fs::remove_file(&resolved).map_err(|source| Error::Io {
        context: format!("delete {}", resolved.display()),
        source,
    })?;
    if let Some(parent) = resolved.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_is_complete_and_never_replaces_an_existing_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/new.txt");
        publish_create_atomic(dir.path(), &path, b"created", None).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"created");
        assert!(publish_create_atomic(dir.path(), &path, b"overwrite", None).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"created");
    }

    #[test]
    fn deletion_requires_the_planned_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.txt");
        std::fs::write(&path, b"other writer").unwrap();
        assert!(publish_delete_atomic(dir.path(), &path, &sha256_hex(b"old")).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"other writer");
        publish_delete_atomic(dir.path(), &path, &sha256_hex(b"other writer")).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn creation_and_deletion_refuse_symlinks_and_escaped_paths() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let external = outside.path().join("new.txt");
        assert!(publish_create_atomic(dir.path(), &external, b"escape", None).is_err());
        assert!(!external.exists());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.path(), dir.path().join("escape")).unwrap();
            assert!(publish_create_atomic(
                dir.path(),
                &dir.path().join("escape/new.txt"),
                b"escape",
                None
            )
            .is_err());
            std::fs::write(&external, b"sentinel").unwrap();
            std::os::unix::fs::symlink(&external, dir.path().join("link.txt")).unwrap();
            assert!(publish_delete_atomic(
                dir.path(),
                &dir.path().join("link.txt"),
                &sha256_hex(b"sentinel")
            )
            .is_err());
            assert_eq!(std::fs::read(&external).unwrap(), b"sentinel");
        }
    }

    #[test]
    fn cas_publishes_when_hash_matches() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, b"before").unwrap();
        let sha = sha256_hex(b"before");
        let new_sha = publish_atomic(dir.path(), &file, b"after", &sha).unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"after");
        assert_eq!(new_sha, sha256_hex(b"after"));
    }

    #[test]
    fn cas_refuses_on_concurrent_change() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, b"before").unwrap();
        let planned_sha = sha256_hex(b"before");
        std::fs::write(&file, b"someone else").unwrap();
        let err = publish_atomic(dir.path(), &file, b"after", &planned_sha);
        assert!(err.is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"someone else");
    }

    #[test]
    fn refuses_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.txt");
        std::fs::write(&real, b"x").unwrap();
        let link = dir.path().join("link.txt");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real, &link).unwrap();
            let err = publish_atomic(dir.path(), &link, b"y", &sha256_hex(b"x"));
            assert!(err.is_err());
        }
    }

    #[test]
    fn refuses_escape() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("v.txt");
        std::fs::write(&victim, b"x").unwrap();
        let err = publish_atomic(dir.path(), &victim, b"y", &sha256_hex(b"x"));
        assert!(err.is_err());
    }

    #[test]
    fn preserves_permissions() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let file = dir.path().join("x.sh");
            std::fs::write(&file, b"#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o755)).unwrap();
            publish_atomic(
                dir.path(),
                &file,
                b"#!/bin/sh\necho hi\n",
                &sha256_hex(b"#!/bin/sh\n"),
            )
            .unwrap();
            let mode = std::fs::metadata(&file).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o755);
        }
    }
}
