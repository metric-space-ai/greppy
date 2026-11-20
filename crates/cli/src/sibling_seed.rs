//! A checkout without an index starts from the index of another clone of the
//! same repository.
//!
//! The first index of a large repository takes minutes and is a heavy host
//! job. A second clone of that repository differs from an indexed one in a
//! few files, so its first index is the other clone's store, copied (a file
//! clone where the file system has them) and then brought to this checkout by
//! the ordinary incremental run. That run compares every file by content, so
//! what is published describes this checkout and nothing else.
//!
//! The copied rows keep the other clone's project column; this checkout
//! adopts that name (`greppy_core::workspace::adopt_project_identity`).

use std::path::{Path, PathBuf};
use std::process::Command;

use greppy_core::error::{Error, Result};
use greppy_core::workspace as workspace_locator;

/// `0` turns the feature off: every first index is built from the sources.
pub(crate) const ENV_SIBLING_SEED: &str = "GREPPY_SIBLING_SEED";

/// How many of the most recently published stores are examined.
const MAX_CANDIDATES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SiblingSeed {
    /// The other clone's published graph database.
    pub(crate) store: PathBuf,
    /// The checkout that store indexes, as its rows spell it.
    pub(crate) root: PathBuf,
    /// The project column of its rows.
    pub(crate) project: String,
}

/// The index this checkout's first index starts from, if another clone of
/// its repository has one. Records the adopted project name, so call it
/// before the project name is read.
pub(crate) fn for_first_index(root: &Path) -> Option<SiblingSeed> {
    let seed = find(root)?;
    if workspace_locator::project_identity(root) != seed.project {
        // A process-wide override outranks the adopted name; under one the
        // copied rows would never be found.
        if std::env::var_os(greppy_core::PROJECT_IDENTITY_ENV).is_some() {
            return None;
        }
        workspace_locator::adopt_project_identity(root, &seed.project).ok()?;
    }
    (workspace_locator::project_identity(root) == seed.project).then_some(seed)
}

fn find(root: &Path) -> Option<SiblingSeed> {
    if std::env::var(ENV_SIBLING_SEED).ok().as_deref() == Some("0") {
        return None;
    }
    // Linked worktrees share an immutable Base instead; a directory without
    // Git has no repository to compare.
    if !root.join(".git").is_dir()
        || has_published_graph(&workspace_locator::store_path(root), root)
    {
        return None;
    }
    let head = git(root, &["rev-parse", "--verify", "--quiet", "HEAD^{commit}"])?;
    let indexer_version =
        greppy_core::indexer_version(&crate::discover_overrides_from_env().ok()?.scope_key());
    let own = greppy_core::workspace::workspace_hash(root);
    let mut stores = greppy_core::cache::workspace_stores()
        .into_iter()
        .filter(|(_, manifest)| manifest.workspace_hash != own)
        .filter_map(|(dir, _)| {
            let graph = dir.join("graph.db");
            let published = std::fs::metadata(&graph).ok()?.modified().ok()?;
            Some((published, graph))
        })
        .collect::<Vec<_>>();
    stores.sort_by_key(|store| std::cmp::Reverse(store.0));
    let mut newest = None;
    for (_, graph) in stores.into_iter().take(MAX_CANDIDATES) {
        let Some((sibling_root, project, sibling_head)) = published_index(&graph, &indexer_version)
        else {
            continue;
        };
        if !same_repository(root, &head, &sibling_root, &sibling_head) {
            continue;
        }
        let seed = SiblingSeed {
            store: graph,
            root: sibling_root,
            project,
        };
        if sibling_head == head {
            return Some(seed);
        }
        newest.get_or_insert(seed);
    }
    newest
}

/// Whether `store` holds an index of `root`. A store that only keeps
/// command-output packs (written before the first query) does not.
pub(crate) fn has_published_graph(store: &Path, root: &Path) -> bool {
    if !store.exists() {
        return false;
    }
    let Ok(store) = greppy_store::Store::open_with(store, greppy_store::OpenOptions::read_only())
    else {
        // Unreadable is not empty: leave it to the ordinary recovery.
        return true;
    };
    !matches!(
        (
            store.get_workspace_state(root.to_string_lossy().as_ref()),
            store.list_projects()
        ),
        (Ok(None), Ok(projects)) if projects.is_empty()
    )
}

/// Checkout root, project name and indexed commit of a store this binary can
/// refresh in place: the whole index of one ordinary checkout, written by
/// the same indexer and schema.
fn published_index(graph: &Path, indexer_version: &str) -> Option<(PathBuf, String, String)> {
    let store =
        greppy_store::Store::open_with(graph, greppy_store::OpenOptions::read_only()).ok()?;
    if store.schema_version().ok()? != greppy_store::migrate::CURRENT_VERSION {
        return None;
    }
    let mut states = store.list_private_workspace_states().ok()?;
    let mut projects = store.list_projects().ok()?;
    if states.len() != 1 || projects.len() != 1 {
        return None;
    }
    let (state, project) = (states.pop()?, projects.pop()?);
    // A linked worktree's store is a private Delta over a shared Base, not a
    // whole index.
    if state.indexer_version != indexer_version
        || state.git_dir.is_none()
        || state.git_dir != state.git_common_dir
        || project.root_path != state.root_path
        || !greppy_core::validate::is_valid_project_name(&project.name)
    {
        return None;
    }
    let head = state.head_oid.filter(|oid| is_object_id(oid))?;
    Some((PathBuf::from(state.root_path), project.name, head))
}

/// Two checkouts belong to one repository when either one has the other's
/// commit.
fn same_repository(root: &Path, head: &str, sibling_root: &Path, sibling_head: &str) -> bool {
    head == sibling_head
        || has_commit(root, sibling_head)
        || (sibling_root.join(".git").is_dir() && has_commit(sibling_root, head))
}

fn has_commit(root: &Path, oid: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["cat-file", "-e"])
        .arg(format!("{oid}^{{commit}}"))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (output.status.success() && is_object_id(value)).then(|| value.to_string())
}

fn is_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn wal_has_frames(graph: &Path) -> bool {
    std::fs::metadata(crate::sqlite_sidecar(graph, "-wal")).is_ok_and(|wal| wal.len() > 0)
}

/// Copy the sibling's published store to `temp_path` and make it this
/// checkout's: its root in the two rows that name one, and this checkout's
/// command-output packs (from `own_store`, if it has any) instead of the
/// other clone's. `Ok(false)`: nothing usable was copied and `temp_path`
/// does not exist.
pub(crate) fn seed_temp_store(
    seed: &SiblingSeed,
    root: &Path,
    own_store: &Path,
    temp_path: &Path,
) -> Result<bool> {
    if !copy_published_store(seed, temp_path)? {
        return Ok(false);
    }
    match take_over(seed, root, own_store, temp_path) {
        Ok(true) => Ok(true),
        outcome => {
            crate::cleanup_sqlite_family(temp_path)?;
            outcome
        }
    }
}

/// The database file alone is the whole published index only while its
/// write-ahead log is empty, and stays so while a reader holds that state:
/// SQLite copies later commits into the file only after every such reader
/// has left. Then the file is cloned. With commits still in the log, SQLite
/// writes the copy itself, from one read snapshot.
fn copy_published_store(seed: &SiblingSeed, temp_path: &Path) -> Result<bool> {
    use rusqlite::OpenFlags;

    // A refresh of the sibling replaces its store by rename; a reader opened
    // before that pins the replaced file, not the one at the path. The copy
    // counts only if the path named one file from before the open until
    // after the copy.
    let pinned_file = file_identity(&seed.store);
    let Ok(reader) = rusqlite::Connection::open_with_flags(
        &seed.store,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return Ok(false);
    };
    let _ = reader.busy_timeout(std::time::Duration::from_secs(15));
    let pinned = reader.execute_batch("BEGIN").is_ok()
        && reader
            .query_row("SELECT count(*) FROM sqlite_master", [], |row| {
                row.get::<_, i64>(0)
            })
            .is_ok();
    if !pinned {
        return Ok(false);
    }
    if wal_has_frames(&seed.store) {
        crate::ensure_copy_headroom(&seed.store, temp_path)?;
        let copied = reader.execute_batch("ROLLBACK").is_ok()
            && reader
                .execute("VACUUM INTO ?1", [temp_path.to_string_lossy()])
                .is_ok();
        if !copied {
            crate::cleanup_sqlite_family(temp_path)?;
            return Ok(false);
        }
    } else if !crate::try_clone_store_file(&seed.store, temp_path)? {
        crate::ensure_copy_headroom(&seed.store, temp_path)?;
        std::fs::copy(&seed.store, temp_path).map_err(|error| {
            Error::io(
                format!(
                    "copy sibling index {} to {}",
                    seed.store.display(),
                    temp_path.display()
                ),
                error,
            )
        })?;
    }
    let same_file = pinned_file.is_some() && pinned_file == file_identity(&seed.store);
    drop(reader);
    if !same_file {
        crate::cleanup_sqlite_family(temp_path)?;
        return Ok(false);
    }
    workspace_locator::ensure_db_mode(temp_path)
        .map_err(|error| Error::io(format!("chmod db {}", temp_path.display()), error))?;
    Ok(true)
}

#[cfg(unix)]
fn file_identity(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path).ok()?;
    Some((metadata.dev(), metadata.ino()))
}

/// Windows does not rename over an open file: the path cannot change under
/// the reader.
#[cfg(not(unix))]
fn file_identity(path: &Path) -> Option<(u64, u64)> {
    path.exists().then_some((0, 0))
}

fn take_over(seed: &SiblingSeed, root: &Path, own_store: &Path, temp_path: &Path) -> Result<bool> {
    let store =
        greppy_store::Store::open_with(temp_path, greppy_store::OpenOptions::query_writer())?;
    let from = seed.root.to_string_lossy();
    let to = root.to_string_lossy();
    let rewrite = || -> rusqlite::Result<bool> {
        let conn = store.conn();
        conn.execute_batch("BEGIN IMMEDIATE")?;
        conn.execute(
            "DELETE FROM workspace_state WHERE root_path <> ?1",
            rusqlite::params![from],
        )?;
        let states = conn.execute(
            "UPDATE workspace_state SET root_path = ?2 WHERE root_path = ?1",
            rusqlite::params![from, to],
        )?;
        let projects = conn.execute(
            "UPDATE projects SET root_path = ?2 WHERE name = ?1",
            rusqlite::params![seed.project, to],
        )?;
        conn.execute("DELETE FROM expand_packs", [])?;
        if states != 1 || projects != 1 {
            conn.execute_batch("ROLLBACK")?;
            return Ok(false);
        }
        conn.execute_batch("COMMIT")?;
        if own_store.exists() {
            // Packs are looked up by id and expire by themselves; a store
            // from another greppy version may not match column for column,
            // and then they are not worth failing the index for.
            conn.execute("ATTACH DATABASE ?1 AS own", [own_store.to_string_lossy()])?;
            let _ = conn.execute(
                "INSERT OR IGNORE INTO expand_packs SELECT * FROM own.expand_packs",
                [],
            );
            conn.execute_batch("DETACH DATABASE own")?;
        }
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(true)
    };
    let taken = rewrite().map_err(|error| {
        Error::Store(format!(
            "take over sibling index {}: {error}",
            temp_path.display()
        ))
    })?;
    drop(store);
    if taken {
        crate::cleanup_sqlite_sidecars(temp_path)?;
    }
    Ok(taken)
}
