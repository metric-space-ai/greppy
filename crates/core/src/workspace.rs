//! Workspace / store locator shared across the passthrough library and the
//! CLI dispatcher.
//!
//! Keeps `.greppy/graph.db` from polluting `grep -R` in the same
//! workspace, and ensures `index <path>` writes to the same location that
//! `search-graph` / `trace` / `search-code` / `semantic` read from.
//! The store directory is created mode 0700 and the DB file mode 0600;
//! symlinked store paths are refused.
//!
//! Rules:
//!
//! 1. The graph DB is **never** placed at `<repo>/.greppy/graph.db`.
//!    That path lives inside the workspace and trips every `grep -R .`
//!    over the SQLite file.
//! 2. Default store location uses the versioned data root:
//!    - `$XDG_DATA_HOME/greppy/workspaces/v2/<ws-hash>/graph.db`
//!      (or `$HOME/.local/share/greppy/...`) on Linux;
//!    - `~/Library/Application Support/greppy/workspaces/v2/<ws-hash>/graph.db`
//!      on macOS;
//!    - `%LOCALAPPDATA%/greppy/workspaces/v2/<ws-hash>/graph.db` on Windows
//!      (Tier 2 — not Tier 1 today; kept so the function compiles).
//! 3. Override via `GREPPY_STORE_DIR=/path/to/dir`; the workspace hash
//!    is still appended so different workspaces do not collide.
//! 4. `<ws-hash>` is the first 16 hex chars of
//!    `sha256(canonical_workspace_root)` — deterministic, not
//!    `DefaultHasher` (which uses a fixed stdlib key and is not stable
//!    across Rust versions).
//! 5. The store dir is created mode 0700; the DB file is created
//!    mode 0600. Existing dirs/files are chmod'd on every call (so a
//!    store created before this rule was applied is also tightened).
//!    Symlinks at either path are refused.
//!
//! Tier 1 today: macOS + Linux. Windows behaviour is documented but not
//! built/tested.

use std::path::{Path, PathBuf};

/// Trusted agent-parent override for the logical project column. Linked and
/// temporary worktrees have unrelated directory basenames; Store-CoW needs
/// all of them to address the same Base rows.
pub const PROJECT_IDENTITY_ENV: &str = "GREPPY_PROJECT_IDENTITY";

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
/// `GREPPY_STORE_DIR` or under the OS cache dir. The directory is
/// **not** created by this function — callers create it on first write
/// so we never leave an empty dir behind on read-only paths.
pub fn store_dir(workspace_root: &Path) -> PathBuf {
    crate::cache::workspace_store_dir(workspace_root)
}

/// Full path to the workspace's graph DB.
pub fn store_path(workspace_root: &Path) -> PathBuf {
    store_dir(workspace_root).join("graph.db")
}

/// Root of Greppy's owned data namespaces.
pub fn store_cache_root() -> Option<PathBuf> {
    Some(crate::cache::data_root())
}

/// Name of the per-store "last used" marker file. Rate-limited to one write
/// per minute and updated on every
/// query that opens the store to serve a read (via [`touch_lastused`]),
/// because a read-only query never bumps `graph.db`'s mtime, so the
/// mtime of the DB alone cannot tell a stale store from a recently used
/// one. [`cleanup_stale_stores`] evicts stores whose marker is older
/// than the TTL.
pub const LASTUSED_MARKER: &str = ".lastused";

/// Touch (create/overwrite) the `.lastused` marker inside `store_dir`
/// to record that the store was used *now*. Best-effort: any I/O error
/// is swallowed — failing to record a touch must never fail a query.
/// The write updates the file's mtime, which is what
/// [`cleanup_stale_stores`] reads.
pub fn touch_lastused(store_dir: &Path) {
    crate::cache::touch_last_used_dir(store_dir);
}

/// Environment variable overriding the store eviction TTL, in **days**.
/// `0` disables only age-based eviction; the quota remains independent.
/// Absent/unparsable falls back to
/// [`STORE_TTL_DAYS_DEFAULT`].
pub const ENV_STORE_TTL_DAYS: &str = "GREPPY_STORE_TTL_DAYS";

/// Default store eviction TTL: **14 days**. A store dir whose
/// `.lastused` marker has not been touched within this window is
/// evicted on the next probabilistic cleanup pass.
pub const STORE_TTL_DAYS_DEFAULT: u64 = 14;

/// Resolve the store eviction TTL in **seconds** from
/// `GREPPY_STORE_TTL_DAYS` (default 14 days). Returns `0` when the env
/// var is `0`, which callers treat as "eviction disabled".
pub fn store_ttl_secs() -> u64 {
    let days = std::env::var(ENV_STORE_TTL_DAYS)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(STORE_TTL_DAYS_DEFAULT);
    days.saturating_mul(24 * 60 * 60)
}

/// Compatibility wrapper around the manifest-verified global GC. The
/// `cache_root` argument is ignored so an arbitrary caller-provided directory
/// can never become a recursive deletion root. `ttl_secs == 0` disables only
/// age eviction; the independently configured quota still applies.
pub fn cleanup_stale_stores(cache_root: &Path, ttl_secs: u64, keep: &Path) -> usize {
    let _ = cache_root; // retained for source compatibility; GC owns its root.
    let current_root = crate::cache::read_store_manifest(keep)
        .ok()
        .map(|m| m.canonical_root);
    let mut policy = crate::cache::GcPolicy::from_env();
    policy.ttl = std::time::Duration::from_secs(ttl_secs);
    crate::cache::run_gc(&policy, false, current_root.as_deref())
        .map(|r| r.removed.len())
        .unwrap_or(0)
}

/// Resolve the canonical workspace root shared by every CLI and passthrough path.
/// `.git` may be either a directory or a file (linked worktree/submodule).
pub fn resolve_workspace_root(start: &Path) -> PathBuf {
    let canonical = start
        .canonicalize()
        .or_else(|_| std::path::absolute(start))
        .unwrap_or_else(|_| start.to_path_buf());
    let mut cur = canonical.as_path();
    let mut nearest_project_marker = None;
    loop {
        // A linked worktree has a `.git` file rather than a directory. In
        // either form it is the authoritative isolation boundary and must
        // win over nested Cargo/Python project markers in a monorepo.
        if cur.join(".git").exists() {
            return cur.to_path_buf();
        }
        if nearest_project_marker.is_none()
            && (cur.join("Cargo.toml").exists() || cur.join("pyproject.toml").exists())
        {
            nearest_project_marker = Some(cur.to_path_buf());
        }
        match cur.parent() {
            Some(parent) if parent != cur => cur = parent,
            _ => return nearest_project_marker.unwrap_or(canonical),
        }
    }
}

/// Return the project-identity string for `start`: the basename of the
/// canonical workspace root resolved by [`resolve_workspace_root`]. Falls
/// back to the basename of `start` itself, then to `"default"` if empty.
///
/// This is the *one* function the CLI dispatcher and the indexer use
/// to derive the `project` column for the store.
/// Using it consistently means a user can `greppy index /path/to/repo`
/// from any cwd, then `greppy search-code Q` from a subdir, and the
/// project identity matches.
pub fn project_identity(start: &Path) -> String {
    if let Ok(identity) = std::env::var(PROJECT_IDENTITY_ENV) {
        let identity = identity.trim();
        if crate::validate::is_valid_project_name(identity) {
            return identity.to_string();
        }
    }
    resolve_workspace_root(start)
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
/// directory is chmod'd to 0700 (so a store created before this
/// hardening is tightened).
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
    crate::cache::secure_private_directory(path)
}

#[cfg(unix)]
fn set_mode_600(path: &Path) -> std::io::Result<()> {
    crate::cache::secure_private_file(path)
}

#[cfg(windows)]
fn set_mode_700(path: &Path) -> std::io::Result<()> {
    crate::cache::secure_private_directory(path)
}

#[cfg(windows)]
fn set_mode_600(path: &Path) -> std::io::Result<()> {
    crate::cache::secure_private_file(path)
}

#[cfg(not(any(unix, windows)))]
fn set_mode_700(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
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
        let _guard = crate::cache::TEST_ENV_LOCK.lock().unwrap();
        // The store directory must NOT be `<root>/.greppy/graph.db`.
        // We assert it is not under the workspace root.
        let tmp = tempdir_root("greppy-locator-test");
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
        let _guard = crate::cache::TEST_ENV_LOCK.lock().unwrap();
        let tmp = tempdir_root("greppy-store-path");
        let sp = store_path(&tmp);
        assert!(sp.ends_with("graph.db"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_store_dir_creates_with_0700() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempdir_root("greppy-ensure-store-dir");
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
        let tmp = tempdir_root("greppy-symlink-store-dir");
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
        let tmp = tempdir_root("greppy-ensure-db-mode");
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
        let tmp = tempdir_root("greppy-projid-git");
        let root = tmp.join("repo");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let nested = root.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(project_identity(&nested), "repo");
        let _ = std::fs::remove_dir_all(&tmp);

        let tmp = tempdir_root("greppy-projid-cargo");
        let root = tmp.join("rustproj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        let nested = root.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(project_identity(&nested), "rustproj");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn git_worktree_root_wins_over_nested_project_marker() {
        let tmp = tempdir_root("greppy-root-nested-project");
        let root = tmp.join("monorepo");
        let nested = root.join("crates/member/src");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            root.join("crates/member/Cargo.toml"),
            "[package]\nname = \"member\"\n",
        )
        .unwrap();

        assert_eq!(
            resolve_workspace_root(&nested),
            root.canonicalize().unwrap()
        );
        assert_eq!(project_identity(&nested), "monorepo");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn project_identity_falls_back_to_basename_when_no_marker() {
        let tmp = tempdir_root("greppy-projid-nomarker");
        let naked = tmp.join("naked/dir");
        std::fs::create_dir_all(&naked).unwrap();
        assert_eq!(project_identity(&naked), "dir");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Back-date `path`'s mtime to `t` via `touch -t` (Tier-1 targets are
    /// Unix; no `filetime` dep needed). Mirrors the helper the sidecar
    /// cleanup tests use.
    fn set_mtime(path: &Path, t: std::time::SystemTime) {
        let secs = t
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let day_secs = secs.rem_euclid(86_400);
        let hour = day_secs / 3600;
        let minute = (day_secs % 3600) / 60;
        let second = day_secs % 60;
        let mut days = secs.div_euclid(86_400);
        let mut year: i64 = 1970;
        loop {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            let year_days = if leap { 366 } else { 365 };
            if days >= year_days {
                days -= year_days;
                year += 1;
            } else {
                break;
            }
        }
        let month_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };
        let mut month = 1;
        for &md in &month_days {
            if days >= md {
                days -= md;
                month += 1;
            } else {
                break;
            }
        }
        let day = days + 1;
        let stamp = format!("{year:04}{month:02}{day:02}{hour:02}{minute:02}.{second:02}");
        let _ = std::process::Command::new("touch")
            .arg("-t")
            .arg(&stamp)
            .arg(path)
            .status();
    }

    /// Write a `.lastused` marker inside `store_dir` and back-date its
    /// mtime by `age_secs` so we can drive `cleanup_stale_stores`
    /// deterministically without sleeping.
    fn write_marker_aged(store_dir: &Path, age_secs: u64) {
        let marker = store_dir.join(LASTUSED_MARKER);
        std::fs::write(&marker, b"").unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs);
        set_mtime(&marker, when);
    }

    #[test]
    fn legacy_cleanup_api_never_removes_unverified_directories() {
        let _guard = crate::cache::TEST_ENV_LOCK.lock().unwrap();
        let cache = tempdir_root("greppy-cleanup-stores");
        let previous_store_dir = std::env::var_os("GREPPY_STORE_DIR");
        std::env::set_var("GREPPY_STORE_DIR", cache.join("managed-data"));
        // An OLD store: marker aged well past the TTL → evicted. Use a
        // large back-date (10 days) so a timezone skew in the `touch -t`
        // helper cannot flip its side of the 1-day cutoff.
        let old = cache.join("aaaaaaaaaaaaaaaa");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("graph.db"), b"sqlite").unwrap();
        write_marker_aged(&old, 10 * 86_400);
        // A FRESH store: marker written just now (not back-dated) →
        // survives. Leaving it at creation time avoids any TZ-skew
        // ambiguity around the cutoff.
        let fresh = cache.join("bbbbbbbbbbbbbbbb");
        std::fs::create_dir_all(&fresh).unwrap();
        std::fs::write(fresh.join("graph.db"), b"sqlite").unwrap();
        touch_lastused(&fresh);
        // The KEEP store: even though its marker is old, it is the store
        // the current invocation is using and must NOT be removed.
        let keep = cache.join("cccccccccccccccc");
        std::fs::create_dir_all(&keep).unwrap();
        std::fs::write(keep.join("graph.db"), b"sqlite").unwrap();
        write_marker_aged(&keep, 10 * 86_400);

        // These directories have no V2 manifest. The compatibility API now
        // delegates to the safe GC and must leave every one untouched.
        let removed = cleanup_stale_stores(&cache, 86_400, &keep);
        assert_eq!(removed, 0, "unverified directories are unmanaged");
        assert!(old.exists(), "old but unverified data must survive");
        assert!(fresh.exists(), "fresh store must survive");
        assert!(keep.exists(), "keep store must survive even when stale");

        // ttl_secs == 0 disables eviction entirely, even for a store
        // whose marker is far past.
        write_marker_aged(&fresh, 10 * 86_400);
        let removed0 = cleanup_stale_stores(&cache, 0, &keep);
        assert_eq!(removed0, 0, "ttl 0 disables eviction");
        assert!(fresh.exists(), "nothing removed when eviction disabled");

        match previous_store_dir {
            Some(path) => std::env::set_var("GREPPY_STORE_DIR", path),
            None => std::env::remove_var("GREPPY_STORE_DIR"),
        }
        let _ = std::fs::remove_dir_all(&cache);
    }

    #[test]
    fn store_ttl_secs_defaults_to_14_days() {
        let _guard = crate::cache::TEST_ENV_LOCK.lock().unwrap();
        // Guard against another test's env mutation leaking in.
        std::env::remove_var(ENV_STORE_TTL_DAYS);
        assert_eq!(store_ttl_secs(), STORE_TTL_DAYS_DEFAULT * 24 * 60 * 60);
        assert_eq!(store_ttl_secs(), 14 * 86_400);
    }

    #[test]
    fn touch_lastused_creates_recent_marker() {
        let dir = tempdir_root("greppy-touch-lastused");
        touch_lastused(&dir);
        let marker = dir.join(LASTUSED_MARKER);
        assert!(marker.exists(), ".lastused marker must be created");
        let mtime = std::fs::metadata(&marker).unwrap().modified().unwrap();
        let age = std::time::SystemTime::now().duration_since(mtime).unwrap();
        assert!(age.as_secs() < 60, "marker mtime must be recent");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
