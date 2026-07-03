//! Workspace / store locator shared across the drop-in wrapper and the
//! CLI dispatcher.
//!
//! Closes R-005 (`.grepplus/graph.db` pollutes `grep -R` from the same
//! workspace) and R-013 (`index <path>` wrote to a different location than
//! `search-graph` / `trace` / `search-code` / `semantic` read from).
//! Also closes the R-019 / R-007 residual: the store directory is created
//! mode 0700 and the DB file mode 0600; symlinked store paths are refused.
//!
//! Rules (2026-06-30, WP-R005 + WP-R013 + RV-007):
//!
//! 1. The graph DB is **never** placed at `<repo>/.grepplus/graph.db`.
//!    That path lives inside the workspace and trips every `grep -R .`
//!    over the SQLite file.
//! 2. Default store location:
//!    - `$XDG_CACHE_HOME/grepplus/<ws-hash>/graph.db` if `XDG_CACHE_HOME`
//!      is set (or `$HOME/.cache/grepplus/...` on Linux);
//!    - `$TMPDIR/grepplus/<ws-hash>/graph.db` as fallback on Unix;
//!    - `%LOCALAPPDATA%/grepplus/<ws-hash>/graph.db` on Windows
//!      (Tier 2 — not Tier 1 today; kept so the function compiles).
//! 3. Override via `GREPPLUS_STORE_DIR=/path/to/dir`; the workspace hash
//!    is still appended so different workspaces do not collide.
//! 4. `<ws-hash>` is the first 16 hex chars of
//!    `sha256(canonical_workspace_root)` — deterministic, not
//!    `DefaultHasher` (which uses a fixed stdlib key and is not stable
//!    across Rust versions; see R-019).
//! 5. The store dir is created mode 0700; the DB file is created
//!    mode 0600. Existing dirs/files are chmod'd on every call (so a
//!    store created before this rule was applied is also tightened).
//!    Symlinks at either path are refused.
//!
//! Tier 1 today: macOS + Linux. Windows behaviour is documented but not
//! built/tested. See R-023.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Canonicalise `path` lexically for the hash (resolve symlinks if
/// possible, otherwise absolute-path it). Best-effort: a path we cannot
/// resolve still gets a hash so writes succeed, just one the user cannot
/// reverse.
fn canonical_for_hash(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Compute the workspace hash from a canonical workspace root.
///
/// The hash is the first 16 hex chars of `sha256(canonical_root)` —
/// deterministic across runs and Rust versions, unlike `DefaultHasher`.
pub fn workspace_hash(workspace_root: &Path) -> String {
    let canon = canonical_for_hash(workspace_root);
    let mut h = Sha256::new();
    h.update(canon.to_string_lossy().as_bytes());
    let digest = h.finalize();
    let hex = format!("{:x}", digest);
    hex.chars().take(16).collect()
}

/// Directory that holds this workspace's graph DB and sidecars, under
/// `GREPPLUS_STORE_DIR` or under the OS cache dir. The directory is
/// **not** created by this function — callers create it on first write
/// so we never leave an empty dir behind on read-only paths.
pub fn store_dir(workspace_root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("GREPPLUS_STORE_DIR") {
        return PathBuf::from(p).join(workspace_hash(workspace_root));
    }

    // macOS / BSD: ~/Library/Caches/grepplus/<ws-hash>
    // Linux XDG:  $XDG_CACHE_HOME/grepplus/<ws-hash>, or ~/.cache/grepplus/...
    // Windows:    %LOCALAPPDATA%\grepplus\<ws-hash>
    // Fallback:   $TMPDIR/grepplus/<ws-hash>.
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            let base = PathBuf::from(home).join("Library").join("Caches");
            return base.join("grepplus").join(workspace_hash(workspace_root));
        }
    }
    #[cfg(all(target_os = "linux", not(target_os = "android")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(xdg)
                .join("grepplus")
                .join(workspace_hash(workspace_root));
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".cache")
                .join("grepplus")
                .join(workspace_hash(workspace_root));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local)
                .join("grepplus")
                .join(workspace_hash(workspace_root));
        }
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        return PathBuf::from(tmp)
            .join("grepplus")
            .join(workspace_hash(workspace_root));
    }
    // Last resort: a per-process tmp dir.
    std::env::temp_dir()
        .join("grepplus")
        .join(workspace_hash(workspace_root))
}

/// Full path to the workspace's graph DB.
pub fn store_path(workspace_root: &Path) -> PathBuf {
    store_dir(workspace_root).join("graph.db")
}

/// Return the project-identity string for `start`: the basename of the
/// canonical repo root (walking up looking for `.git`, `Cargo.toml`,
/// or `pyproject.toml`). Falls back to the basename of `start` itself,
/// then to `"default"` if the basename is empty.
///
/// This is the *one* function the CLI dispatcher and the indexer use
/// to derive the `project` column for the store — RV-011 / WP-R013.
/// Using it consistently means a user can `grepplus index /path/to/repo`
/// from any cwd, then `grepplus search-code Q` from a subdir, and the
/// project identity matches.
pub fn project_identity(start: &Path) -> String {
    let canonical = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    let mut cur = canonical.clone();
    let mut found_marker = false;
    loop {
        if cur.join(".git").exists()
            || cur.join("Cargo.toml").exists()
            || cur.join("pyproject.toml").exists()
        {
            found_marker = true;
            break;
        }
        match cur.parent() {
            Some(p) if !p.as_os_str().is_empty() && p != cur => cur = p.to_path_buf(),
            _ => break,
        }
    }
    // When no marker exists in the chain, return the basename of the
    // original `start` (not the walked-up `cur`, which would be `/`
    // or `private`).
    let final_path = if found_marker { &cur } else { &canonical };
    final_path
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default".to_string())
}

/// Mode for newly-created store/sidecar directories. The actual `mkdir`
/// lives in [`ensure_store_dir`] and the DB-mode chmod in
/// [`ensure_db_mode`] — both are wired from callers so the helper
/// isn't dead code.
#[deprecated(note = "use ensure_store_dir / ensure_db_mode instead")]
pub fn dir_mode_default() -> u32 {
    0o700
}

/// Create the store directory at `dir` (and any missing parents),
/// mode 0700. Refuses to operate on a symlink. Idempotent: an existing
/// directory is chmod'd to 0700 (so a store created before the
/// 2026-06-30 hardening is tightened). RV-007 / WP-R019.
pub fn ensure_store_dir(dir: &Path) -> std::io::Result<()> {
    if let Ok(md) = std::fs::symlink_metadata(dir) {
        if md.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("refusing symlink at store dir {}", dir.display()),
            ));
        }
        set_mode_700(dir)?;
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    set_mode_700(dir)
}

/// Set the DB file at `path` to mode 0600. Refuses to operate on a
/// symlink. No-op if the file does not yet exist (the writer creates
/// it with the right mode via `OpenOptions::mode`).
pub fn ensure_db_mode(path: &Path) -> std::io::Result<()> {
    let md = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    if md.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing symlink at db path {}", path.display()),
        ));
    }
    set_mode_600(path)
}

#[cfg(unix)]
fn set_mode_700(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
}

#[cfg(unix)]
fn set_mode_600(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_mode_700(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn set_mode_600(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_hash_is_deterministic_and_changes_with_path() {
        let h1 = workspace_hash(Path::new("/tmp/repo-a"));
        let h2 = workspace_hash(Path::new("/tmp/repo-a"));
        let h3 = workspace_hash(Path::new("/tmp/repo-b"));
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 16);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn store_dir_never_returns_path_inside_workspace_root() {
        // The store directory must NOT be `<root>/.grepplus/graph.db`
        // (R-005). We assert it is not under the workspace root.
        let tmp = tempdir_root("grepplus-locator-test");
        let d = store_dir(&tmp);
        assert!(
            !d.starts_with(&tmp),
            "store_dir {d:?} must not be inside workspace root {tmp:?}"
        );
        // And it must include a workspace hash segment.
        let tail = d.file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(tail.len(), 16, "store_dir tail must be a 16-char hex hash");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn store_path_is_store_dir_plus_graph_db() {
        let tmp = tempdir_root("grepplus-store-path");
        let sp = store_path(&tmp);
        assert!(sp.ends_with("graph.db"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_store_dir_creates_with_0700() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempdir_root("grepplus-ensure-store-dir");
        let new_dir = tmp.join("store");
        ensure_store_dir(&new_dir).unwrap();
        let md = std::fs::metadata(&new_dir).unwrap();
        assert_eq!(
            md.permissions().mode() & 0o777,
            0o700,
            "store dir must be 0700; got {:o}",
            md.permissions().mode() & 0o777
        );
        // Idempotent chmod of a pre-existing 0755 dir.
        std::fs::set_permissions(&new_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        ensure_store_dir(&new_dir).unwrap();
        let md = std::fs::metadata(&new_dir).unwrap();
        assert_eq!(
            md.permissions().mode() & 0o777,
            0o700,
            "ensure_store_dir must re-tighten to 0700 even when pre-existing"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_store_dir_refuses_symlink() {
        let tmp = tempdir_root("grepplus-symlink-store-dir");
        let real = tmp.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = tmp.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let r = ensure_store_dir(&link);
        assert!(r.is_err(), "must refuse symlinked store dir");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_db_mode_chmods_to_0600_and_refuses_symlink() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempdir_root("grepplus-ensure-db-mode");
        let db = tmp.join("graph.db");
        std::fs::write(&db, b"sqlite").unwrap();
        std::fs::set_permissions(&db, std::fs::Permissions::from_mode(0o644)).unwrap();
        ensure_db_mode(&db).unwrap();
        let md = std::fs::metadata(&db).unwrap();
        assert_eq!(
            md.permissions().mode() & 0o777,
            0o600,
            "DB must be 0600 after ensure_db_mode; got {:o}",
            md.permissions().mode() & 0o777
        );
        // Symlink refusal.
        let link = tmp.join("graph.db.link");
        std::os::unix::fs::symlink(&db, &link).unwrap();
        let r = ensure_db_mode(&link);
        assert!(r.is_err(), "must refuse symlinked DB");
        // Non-existent file: silent no-op (writer will create with the
        // right mode via OpenOptions::mode).
        let phantom = tmp.join("nope.db");
        ensure_db_mode(&phantom).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn tempdir_root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn project_identity_walks_up_to_git_or_cargo() {
        let tmp = tempdir_root("grepplus-projid-git");
        let root = tmp.join("repo");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let nested = root.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(project_identity(&nested), "repo");
        let _ = std::fs::remove_dir_all(&tmp);

        let tmp = tempdir_root("grepplus-projid-cargo");
        let root = tmp.join("rustproj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        let nested = root.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(project_identity(&nested), "rustproj");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn project_identity_falls_back_to_basename_when_no_marker() {
        let tmp = tempdir_root("grepplus-projid-nomarker");
        let naked = tmp.join("naked/dir");
        std::fs::create_dir_all(&naked).unwrap();
        assert_eq!(project_identity(&naked), "dir");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
