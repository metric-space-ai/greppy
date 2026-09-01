//! Store-CoW lifecycle shared by `greppy -p`, index warming, and query opens.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use greppy_core::error::{Error, Result};
use greppy_store::{BaseBuilderLease, BaseStoreIdentity, BaseStoreLayout, VisibilityIndex};
use sha2::{Digest, Sha256};

pub(crate) const ENV_MODE: &str = "GREPPY_AGENT_STORE_MODE";
pub(crate) const ENV_BASE_PATH: &str = "GREPPY_AGENT_BASE_STORE";
pub(crate) const ENV_BASE_COMMIT: &str = "GREPPY_AGENT_BASE_COMMIT";
pub(crate) const ENV_BASE_REUSED: &str = "GREPPY_AGENT_BASE_REUSED";
pub(crate) const ENV_FALLBACK_REASON: &str = "GREPPY_AGENT_STORE_FALLBACK_REASON";
pub(crate) const ENV_DISABLE_AUTO_LINKED_WORKTREE: &str = "GREPPY_DISABLE_AUTO_LINKED_WORKTREE_COW";
pub(crate) const MODE_OVERLAY: &str = "overlay";
pub(crate) const MODE_PRIVATE: &str = "private";
const VISIBILITY_META_KEY: &str = "store_cow.visibility.v1";
const OVERLAY_BINDING_META_KEY: &str = "store_cow.binding.v1";
const BASE_BUILDER_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
#[cfg(debug_assertions)]
const ENV_TEST_BASE_SUMMARY_FAIL: &str = "GREPPY_TEST_BASE_SUMMARY_FAIL";
#[cfg(debug_assertions)]
const ENV_TEST_FORBID_TEMP_BASE_CHECKOUT: &str = "GREPPY_TEST_FORBID_TEMP_BASE_CHECKOUT";

#[derive(Debug, Clone)]
pub(crate) struct OverlaySpec {
    pub base_path: PathBuf,
    pub visibility: VisibilityIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OverlayFreshnessProof {
    Fresh {
        total_inventory: usize,
    },
    Stale {
        changed_paths: Vec<String>,
        reason: String,
    },
}

#[derive(Debug)]
pub(crate) struct PreparedBase {
    pub graph_path: PathBuf,
    pub identity_hash: String,
    pub reused: bool,
    _reader_lease: greppy_store::BaseReaderLease,
}

/// Keeps the clean Base workspace and its reader lease alive for the complete
/// linked-worktree Delta publication. Environment changes are command-scoped
/// and restored for in-process tests.
pub(crate) struct AutoLinkedWorktreeOverlay {
    _prepared: PreparedBase,
    restore: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl Drop for AutoLinkedWorktreeOverlay {
    fn drop(&mut self) {
        for (name, value) in self.restore.drain(..).rev() {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

pub(crate) fn overlay_spec(root: &Path) -> Result<Option<OverlaySpec>> {
    overlay_spec_inner(root, true)
}

/// Resolve overlay state for a writer. Unlike steady-state query opens, an
/// index refresh must observe Git live so arbitrary edits made outside greppy
/// become part of the next atomic Delta generation.
pub(crate) fn overlay_spec_live(root: &Path) -> Result<Option<OverlaySpec>> {
    overlay_spec_inner(root, false)
}

fn overlay_spec_inner(root: &Path, allow_cached_visibility: bool) -> Result<Option<OverlaySpec>> {
    let Some((base_path, base_commit)) = overlay_environment(root)? else {
        return Ok(None);
    };
    let visibility = if allow_cached_visibility {
        cached_visibility(root, &base_commit)
            .unwrap_or_else(|| visibility_against(root, &base_commit))?
    } else {
        visibility_against(root, &base_commit)?
    };
    Ok(Some(OverlaySpec {
        base_path,
        visibility,
    }))
}

pub(crate) fn overlay_environment(root: &Path) -> Result<Option<(PathBuf, String)>> {
    if let Ok(mode) = std::env::var(ENV_MODE) {
        if mode != MODE_OVERLAY {
            return Ok(None);
        }
        let base_path = std::env::var_os(ENV_BASE_PATH)
            .map(PathBuf::from)
            .ok_or_else(|| Error::Invalid(format!("{ENV_MODE}=overlay without {ENV_BASE_PATH}")))?;
        if !base_path.is_file() {
            return Err(Error::Invalid(format!(
                "configured immutable Base Store is missing: {}; run `greppy index` to rebuild the linked-worktree Base",
                base_path.display()
            )));
        }
        let base_commit = std::env::var(ENV_BASE_COMMIT)
            .map_err(|_| Error::Invalid(format!("{ENV_MODE}=overlay without {ENV_BASE_COMMIT}")))?;
        return Ok(Some((base_path, base_commit)));
    }

    let delta_path = crate::workspace_locator::store_path(root);
    if !delta_path.is_file() {
        return Ok(None);
    }
    let Ok(connection) = rusqlite::Connection::open_with_flags(
        &delta_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        // This lookup runs before the normal snapshot integrity/recovery path.
        // A corrupt or legacy active DB cannot contain a trustworthy binding;
        // let the indexer quarantine/replace it instead of blocking recovery.
        return Ok(None);
    };
    let raw = match connection.query_row(
        "SELECT value FROM schema_meta WHERE key = ?1",
        [OVERLAY_BINDING_META_KEY],
        |row| row.get::<_, String>(0),
    ) {
        Ok(raw) => raw,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
        // Missing schema_meta (old snapshot) and corrupt SQLite are handled by
        // the ordinary index integrity path. Neither is evidence of a usable
        // persisted overlay binding.
        Err(_) => return Ok(None),
    };
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
        Error::Invalid(format!("decode linked-worktree Delta binding: {error}"))
    })?;
    if value.get("version").and_then(serde_json::Value::as_u64) != Some(1) {
        return Err(Error::Invalid(
            "unsupported linked-worktree Delta binding version; run `greppy index` to rebuild it"
                .into(),
        ));
    }
    let base_path = value
        .get("base_path")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Invalid("linked-worktree Delta binding lacks base_path".into()))?;
    if !base_path.is_file() {
        return Err(Error::Invalid(format!(
            "linked-worktree Base Store is missing: {}; run `greppy index` to rebuild it",
            base_path.display()
        )));
    }
    let base_commit = value
        .get("base_commit")
        .and_then(serde_json::Value::as_str)
        .filter(|commit| {
            matches!(commit.len(), 40 | 64) && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        .ok_or_else(|| {
            Error::Invalid("linked-worktree Delta binding has invalid base_commit".into())
        })?
        .to_string();
    let project = value
        .get("project")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::Invalid("linked-worktree Delta binding lacks project".into()))?;
    let expected_project = greppy_core::project_identity(root);
    if project != expected_project {
        return Err(Error::Invalid(format!(
            "linked-worktree Base project mismatch: binding `{project}`, workspace `{expected_project}`; run `greppy index` to rebuild the binding"
        )));
    }
    Ok(Some((base_path, base_commit)))
}

/// Prove freshness from the immutable Base plus the small private Delta.
///
/// A Store-CoW query must not re-walk and stat/hash the complete repository:
/// the published Base already binds an exact Git tree and the Delta binding
/// persists the paths hidden from that Base. We still fail closed. The fast
/// proof verifies the published Base bytes, its tree and project identity,
/// compares the persisted visibility with live Git, and validates the current
/// contents of every dirty path against the private Delta. Any shape we do not
/// understand returns `None` so the ordinary full-inventory check remains the
/// fallback.
pub(crate) fn overlay_freshness_proof(
    root: &Path,
    store: &greppy_store::Store,
    project: &str,
) -> Result<Option<OverlayFreshnessProof>> {
    if !store.is_overlay() {
        return Ok(None);
    }
    let Some((base_path, base_commit)) = overlay_environment(root)? else {
        return Ok(None);
    };
    let Some(cached) = cached_visibility(root, &base_commit) else {
        return Ok(None);
    };
    let cached = cached?;
    let live = visibility_against(root, &base_commit)?;
    if cached != live {
        return Ok(Some(OverlayFreshnessProof::Stale {
            changed_paths: visibility_changed_paths(&cached, &live),
            reason: "live Git changes differ from the indexed Store-CoW Delta".into(),
        }));
    }

    let attached = store.overlay_base_path().ok_or_else(|| {
        Error::Invalid("Store reports overlay mode without an attached Base".into())
    })?;
    if !paths_resolve_equal(attached, &base_path) {
        return Err(Error::Invalid(format!(
            "attached Base {} differs from bound Base {}",
            attached.display(),
            base_path.display()
        )));
    }
    let manifest = verified_manifest_for_graph(&base_path)
        .map_err(|error| Error::io("verify immutable Store-CoW Base", error))?;
    let tree_expr = format!("{base_commit}^{{tree}}");
    let live_base_tree = git_output(root, &["rev-parse", &tree_expr])?;
    if live_base_tree != manifest.identity.base_tree_oid {
        return Err(Error::Invalid(format!(
            "Store-CoW Base tree mismatch: binding {live_base_tree}, manifest {}",
            manifest.identity.base_tree_oid
        )));
    }
    if manifest.identity.store_schema_version != greppy_store::migrate::CURRENT_VERSION
        || manifest.identity.indexer_version != greppy_core::INDEXER_VERSION_BASE
        || manifest.identity.parser_and_extractor_versions
            != format!("greppy-parser/extractor-{}", env!("CARGO_PKG_VERSION"))
    {
        return Ok(None);
    }

    let base_project_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM greppy_base.projects WHERE name = ?1",
            [project],
            |row| row.get(0),
        )
        .map_err(|error| Error::Store(format!("verify Store-CoW Base project: {error}")))?;
    if base_project_count != 1 {
        return Err(Error::Invalid(format!(
            "Store-CoW Base project mismatch: expected exactly `{project}`"
        )));
    }
    let root_string = root.to_string_lossy();
    let Some(workspace) = store
        .get_workspace_state(&root_string)
        .map_err(|error| Error::Store(format!("read Store-CoW workspace state: {error}")))?
    else {
        return Ok(None);
    };
    if workspace.schema_version != greppy_store::migrate::CURRENT_VERSION
        || workspace.indexer_version != greppy_core::INDEXER_VERSION_BASE
        || workspace.graph_generation == 0
    {
        return Ok(None);
    }

    let dirty = cached
        .dirty_paths()
        .collect::<std::collections::BTreeSet<_>>();
    let deleted = cached
        .deleted_paths()
        .collect::<std::collections::BTreeSet<_>>();
    for rel_path in &deleted {
        match std::fs::symlink_metadata(root.join(rel_path)) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Ok(Some(OverlayFreshnessProof::Stale {
                    changed_paths: vec![(*rel_path).to_owned()],
                    reason: "a path recorded as deleted exists again".into(),
                }));
            }
            Err(error) => {
                return Err(Error::io(
                    format!("stat deleted Store-CoW path {rel_path}"),
                    error,
                ));
            }
        }
    }

    let identities = store
        .list_file_identities(project)
        .map_err(|error| Error::Store(format!("read Store-CoW file identities: {error}")))?;
    for rel_path in &dirty {
        if !persisted_delta_path_matches(root, store, project, rel_path, &identities)? {
            return Ok(Some(OverlayFreshnessProof::Stale {
                changed_paths: vec![(*rel_path).to_owned()],
                reason: "a Store-CoW Delta path changed after it was indexed".into(),
            }));
        }
    }

    let private_paths = private_delta_paths(store)?;
    if let Some(unbound) = private_paths
        .iter()
        .find(|path| !dirty.contains(path.as_str()))
    {
        return Err(Error::Invalid(format!(
            "private Store-CoW row `{unbound}` is absent from the Delta visibility manifest"
        )));
    }

    let total_inventory = store
        .file_count(project)
        .map_err(|error| Error::Store(format!("count Store-CoW inventory: {error}")))?;
    let total_inventory = usize::try_from(total_inventory)
        .map_err(|_| Error::Invalid("Store-CoW inventory count is negative".into()))?;
    Ok(Some(OverlayFreshnessProof::Fresh { total_inventory }))
}

fn visibility_changed_paths(cached: &VisibilityIndex, live: &VisibilityIndex) -> Vec<String> {
    let cached_dirty = cached
        .dirty_paths()
        .collect::<std::collections::BTreeSet<_>>();
    let live_dirty = live
        .dirty_paths()
        .collect::<std::collections::BTreeSet<_>>();
    let cached_deleted = cached
        .deleted_paths()
        .collect::<std::collections::BTreeSet<_>>();
    let live_deleted = live
        .deleted_paths()
        .collect::<std::collections::BTreeSet<_>>();
    cached_dirty
        .symmetric_difference(&live_dirty)
        .chain(cached_deleted.symmetric_difference(&live_deleted))
        .map(|path| (*path).to_owned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn private_delta_paths(store: &greppy_store::Store) -> Result<std::collections::BTreeSet<String>> {
    let mut private_paths = std::collections::BTreeSet::new();
    for query in [
        "SELECT rel_path FROM main.file_state",
        "SELECT rel_path FROM main.index_skips",
        "SELECT file_path FROM main.nodes WHERE file_path <> '' AND label <> 'Folder'",
        "SELECT file_path FROM main.raw_edges WHERE file_path <> ''",
        "SELECT rel_path FROM main.file_content WHERE rel_path <> ''",
        "SELECT file_path FROM main.vector_embeddings WHERE file_path <> ''",
    ] {
        let mut statement = store
            .conn()
            .prepare(query)
            .map_err(|error| Error::Store(format!("inspect Store-CoW Delta paths: {error}")))?;
        let paths = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| Error::Store(format!("query Store-CoW Delta paths: {error}")))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|error| Error::Store(format!("read Store-CoW Delta paths: {error}")))?;
        private_paths.extend(paths);
    }
    Ok(private_paths)
}

fn persisted_delta_path_matches(
    root: &Path,
    store: &greppy_store::Store,
    project: &str,
    rel_path: &str,
    identities: &std::collections::HashMap<String, greppy_store::FileIdentity>,
) -> Result<bool> {
    let path = root.join(rel_path);
    let metadata = std::fs::metadata(&path)
        .map_err(|error| Error::io(format!("stat Store-CoW Delta path {rel_path}"), error))?;
    if !metadata.is_file() {
        return Ok(false);
    }
    let current = greppy_discover::stable_metadata(&metadata);
    if let Some(state) = store
        .get_file_state(project, rel_path)
        .map_err(|error| Error::Store(format!("read Store-CoW file state: {error}")))?
    {
        let identity = identities.get(rel_path);
        let stat_matches = state.size >= 0
            && state.size as u64 == current.size
            && current.mtime_ns == Some(state.mtime_ns)
            && identity.is_some_and(|identity| {
                identity.ctime_ns == current.ctime_ns && identity.file_id == current.file_id
            });
        if stat_matches {
            return Ok(true);
        }
        if current.size > greppy_freshness::incremental::MAX_FILE_SIZE_BYTES {
            return Ok(false);
        }
        let (bytes, _) = greppy_discover::read_stable_file(&path)
            .map_err(|error| Error::io(format!("read Store-CoW Delta path {rel_path}"), error))?;
        return Ok(greppy_store::file_state::sha256_hex(&bytes) == state.sha256);
    }
    if let Some(skip) = store
        .get_index_skip(project, rel_path)
        .map_err(|error| Error::Store(format!("read Store-CoW skip state: {error}")))?
    {
        return Ok(skip.size >= 0
            && skip.size as u64 == current.size
            && current.mtime_ns == Some(skip.mtime_ns)
            && skip.ctime_ns == current.ctime_ns
            && skip.file_id == current.file_id);
    }
    Ok(false)
}

fn paths_resolve_equal(left: &Path, right: &Path) -> bool {
    left == right
        || left.canonicalize().ok() == right.canonicalize().ok()
        || left == right.canonicalize().unwrap_or_else(|_| right.to_path_buf())
}

fn cached_visibility(root: &Path, base_commit: &str) -> Option<Result<VisibilityIndex>> {
    let path = crate::workspace_locator::store_path(root);
    if !path.is_file() {
        return None;
    }
    let connection = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    cached_visibility_from_connection(&connection, base_commit)
}

pub(crate) fn cached_visibility_from_connection(
    connection: &rusqlite::Connection,
    base_commit: &str,
) -> Option<Result<VisibilityIndex>> {
    let raw = connection
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [VISIBILITY_META_KEY],
            |row| row.get::<_, String>(0),
        )
        .ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    if value.get("base_commit").and_then(serde_json::Value::as_str) != Some(base_commit) {
        return None;
    }
    let paths = |key: &str| -> Option<Vec<String>> {
        value
            .get(key)?
            .as_array()?
            .iter()
            .map(|item| item.as_str().map(ToOwned::to_owned))
            .collect()
    };
    let dirty = paths("dirty")?;
    let deleted = paths("deleted")?;
    Some(
        VisibilityIndex::new(dirty, deleted)
            .map_err(|error| Error::io("validate cached Store Delta visibility", error)),
    )
}

pub(crate) fn visibility_for_open_connection(
    root: &Path,
    base_commit: &str,
    connection: &rusqlite::Connection,
) -> Result<VisibilityIndex> {
    cached_visibility_from_connection(connection, base_commit)
        .unwrap_or_else(|| visibility_against(root, base_commit))
}

pub(crate) fn persist_visibility(
    store: &greppy_store::Store,
    visibility: &VisibilityIndex,
    base_commit: &str,
) -> Result<()> {
    let value = serde_json::json!({
        "base_commit": base_commit,
        "dirty": visibility.dirty_paths().collect::<Vec<_>>(),
        "deleted": visibility.deleted_paths().collect::<Vec<_>>(),
    });
    let raw = serde_json::to_string(&value)
        .map_err(|error| Error::Invalid(format!("serialize Store Delta visibility: {error}")))?;
    store
        .conn()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES(?1, ?2)\n             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (VISIBILITY_META_KEY, raw),
        )
        .map_err(|error| Error::Store(format!("persist Store Delta visibility: {error}")))?;
    Ok(())
}

pub(crate) fn persist_overlay_binding(
    store: &greppy_store::Store,
    base_path: &Path,
    base_commit: &str,
    project: &str,
) -> Result<()> {
    let value = serde_json::json!({
        "version": 1,
        "base_path": base_path,
        "base_commit": base_commit,
        "project": project,
    });
    let raw = serde_json::to_string(&value)
        .map_err(|error| Error::Invalid(format!("serialize Store-CoW binding: {error}")))?;
    store
        .conn()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES(?1, ?2)\n             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (OVERLAY_BINDING_META_KEY, raw),
        )
        .map_err(|error| Error::Store(format!("persist Store-CoW binding: {error}")))?;
    Ok(())
}

pub(crate) fn configure_overlay_environment(prepared: &PreparedBase, base_commit: &str) {
    std::env::set_var(ENV_MODE, MODE_OVERLAY);
    std::env::set_var(ENV_BASE_PATH, &prepared.graph_path);
    std::env::set_var(ENV_BASE_COMMIT, base_commit);
    std::env::set_var(ENV_BASE_REUSED, if prepared.reused { "1" } else { "0" });
    std::env::remove_var(ENV_FALLBACK_REASON);
}

pub(crate) fn configure_private_environment(reason: &str) {
    clear_overlay_environment();
    std::env::set_var(ENV_MODE, MODE_PRIVATE);
    std::env::set_var(ENV_FALLBACK_REASON, reason);
}

pub(crate) fn clear_overlay_environment() {
    std::env::remove_var(ENV_MODE);
    std::env::remove_var(ENV_BASE_PATH);
    std::env::remove_var(ENV_BASE_COMMIT);
    std::env::remove_var(ENV_BASE_REUSED);
    std::env::remove_var(ENV_FALLBACK_REASON);
}

pub(crate) fn diagnostics(
    root: &Path,
    store: &greppy_store::Store,
    delta_path: &Path,
) -> serde_json::Value {
    diagnostics_inner(root, Some(store), delta_path)
}

pub(crate) fn diagnostics_without_store(root: &Path, delta_path: &Path) -> serde_json::Value {
    diagnostics_inner(root, None, delta_path)
}

fn diagnostics_inner(
    root: &Path,
    store: Option<&greppy_store::Store>,
    delta_path: &Path,
) -> serde_json::Value {
    let resolved_binding = overlay_environment(root);
    let binding_is_persisted = std::env::var_os(ENV_MODE).is_none();
    let configured_mode = std::env::var(ENV_MODE).ok().or_else(|| {
        resolved_binding
            .as_ref()
            .ok()
            .and_then(|binding| binding.as_ref())
            .map(|_| MODE_OVERLAY.to_string())
    });
    let fallback_reason = std::env::var(ENV_FALLBACK_REASON).ok();
    let base_commit = std::env::var(ENV_BASE_COMMIT).ok().or_else(|| {
        resolved_binding
            .as_ref()
            .ok()
            .and_then(|binding| binding.as_ref())
            .map(|(_, commit)| commit.clone())
    });
    let base_reused = std::env::var(ENV_BASE_REUSED)
        .ok()
        .as_deref()
        .map(|value| value == "1")
        .or_else(|| {
            binding_is_persisted
                .then(|| {
                    resolved_binding
                        .as_ref()
                        .ok()
                        .and_then(|binding| binding.as_ref())
                        .map(|_| true)
                })
                .flatten()
        });

    let mut base_path = None;
    let mut base_identity = None;
    let mut base_complete = None;
    let mut dirty_paths = None;
    let mut deleted_paths = None;
    let mut delta_identity = None;
    let mut error = resolved_binding.err().map(|issue| issue.to_string());
    if configured_mode.as_deref() == Some(MODE_OVERLAY) {
        match overlay_spec(root) {
            Ok(Some(spec)) => {
                dirty_paths = Some(spec.visibility.dirty_paths().count());
                deleted_paths = Some(spec.visibility.deleted_paths().count());
                base_path = Some(spec.base_path.clone());
                match verified_manifest_for_graph(&spec.base_path) {
                    Ok(manifest) => {
                        base_identity = Some(manifest.identity_hash.clone());
                        base_complete = Some(true);
                        let identity_payload = serde_json::json!({
                            "base_identity": manifest.identity_hash,
                            "base_commit": base_commit,
                            "dirty": spec.visibility.dirty_paths().collect::<Vec<_>>(),
                            "deleted": spec.visibility.deleted_paths().collect::<Vec<_>>(),
                        });
                        if let Ok(bytes) = serde_json::to_vec(&identity_payload) {
                            delta_identity = Some(hex_sha256(&bytes));
                        }
                    }
                    Err(issue) => {
                        base_complete = Some(false);
                        error = Some(issue.to_string());
                    }
                }
            }
            Ok(None) => {}
            Err(issue) => error = Some(issue.to_string()),
        }
    }

    let count = |table: &str| -> Option<i64> {
        store?
            .conn()
            .query_row(&format!("SELECT COUNT(*) FROM main.{table}"), [], |row| {
                row.get(0)
            })
            .ok()
    };
    serde_json::json!({
        "mode": configured_mode.as_deref().unwrap_or(if store.is_some_and(greppy_store::Store::is_overlay) { MODE_OVERLAY } else { "single" }),
        "base_path": base_path,
        "base_identity": base_identity,
        "base_commit": base_commit,
        "base_complete": base_complete,
        "base_cache_hit": base_reused,
        "delta_path": delta_path,
        "delta_identity": delta_identity,
        "dirty_file_count": dirty_paths,
        "deleted_file_count": deleted_paths,
        "delta_rows": {
            "nodes": count("nodes"),
            "raw_edges": count("raw_edges"),
            "edges": count("overlay_edges"),
            "file_content": count("file_content"),
            "embeddings": count("vector_embeddings"),
        },
        "fallback_reason": fallback_reason,
        "error": error,
    })
}

fn verified_manifest_for_graph(path: &Path) -> std::io::Result<greppy_store::BaseStoreManifest> {
    let identity_dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("Base graph has no identity directory"))?;
    let repo_dir = identity_dir
        .parent()
        .ok_or_else(|| std::io::Error::other("Base graph has no repository directory"))?;
    let version_dir = repo_dir
        .parent()
        .ok_or_else(|| std::io::Error::other("Base graph has no format directory"))?;
    let stores_dir = version_dir
        .parent()
        .ok_or_else(|| std::io::Error::other("Base graph has no stores directory"))?;
    let data_root = stores_dir
        .parent()
        .ok_or_else(|| std::io::Error::other("Base graph has no data root"))?;
    let bytes = std::fs::read(identity_dir.join(greppy_store::BASE_STORE_MANIFEST_FILE))?;
    let manifest: greppy_store::BaseStoreManifest = serde_json::from_slice(&bytes)
        .map_err(|issue| std::io::Error::other(format!("decode Base manifest: {issue}")))?;
    let layout = BaseStoreLayout::new(data_root, &manifest.identity)?;
    if layout.graph != path {
        return Err(std::io::Error::other(
            "Base graph path does not match identity",
        ));
    }
    layout.read_verified_manifest()
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn report_base_phase(path: Option<&Path>, phase: &str) {
    let Some(path) = path else { return };
    let Some(mut job) = crate::read_background_job(path) else {
        return;
    };
    job["state"] = serde_json::json!(phase);
    job["updated_at_unix_secs"] = serde_json::json!(crate::unix_now_secs_cli());
    job["completed_spans"] = serde_json::json!(0);
    job["total_spans"] = serde_json::json!(0);
    job["progress_milli_percent"] = serde_json::json!(0);
    job["progress_unit"] = serde_json::json!("steps");
    job["rate_milli_spans_per_second"] = serde_json::Value::Null;
    job["eta_seconds"] = serde_json::Value::Null;
    job["eta_minutes"] = serde_json::Value::Null;
    job["eta_unix_secs"] = serde_json::Value::Null;
    job["last_error"] = serde_json::Value::Null;
    let _ = crate::write_background_job(path, &job);
}

fn acquire_base_builder(
    layout: &BaseStoreLayout,
    identity_hash: &str,
    progress_path: Option<&Path>,
    max_wait: std::time::Duration,
) -> Result<BaseBuilderLease> {
    let started = std::time::Instant::now();
    loop {
        if let Some(lease) = layout
            .acquire_builder(true)
            .map_err(|error| Error::io("acquire Base Store builder lease", error))?
        {
            return Ok(lease);
        }
        report_base_phase(progress_path, "waiting_for_base_builder");
        if started.elapsed() >= max_wait {
            let lock_path = layout
                .builder_lock_path()
                .map_err(|error| Error::io("resolve Base builder lock", error))?;
            return Err(Error::Lock(format!(
                "another worktree is building immutable Base {identity_hash}; lock {}; wait for that build to publish, then rerun `greppy index`",
                lock_path.display()
            )));
        }
        std::thread::sleep(std::time::Duration::from_millis(250).min(max_wait));
    }
}

/// Prepare an immutable Base from the primary checkout and attach the current
/// linked Git worktree as a private Delta. The primary checkout's HEAD is the
/// repository-wide pinned Base; committed branch differences, dirty files and
/// untracked files are all represented by [`visibility_against`].
pub(crate) fn prepare_auto_linked_worktree_overlay(
    root: &Path,
    shared_data_root: &Path,
    embedding_args: crate::EmbeddingCliArgs<'_>,
    progress_path: Option<&Path>,
) -> Result<Option<AutoLinkedWorktreeOverlay>> {
    if std::env::var(ENV_DISABLE_AUTO_LINKED_WORKTREE)
        .ok()
        .as_deref()
        == Some("1")
        || std::env::var_os(ENV_MODE).is_some()
        || !root.join(".git").is_file()
    {
        return Ok(None);
    }
    let primary = primary_worktree_root(root)?;
    let project = greppy_core::project_identity(&primary);
    let names = [
        greppy_core::PROJECT_IDENTITY_ENV,
        ENV_MODE,
        ENV_BASE_PATH,
        ENV_BASE_COMMIT,
        ENV_BASE_REUSED,
        ENV_FALLBACK_REASON,
        ENV_DISABLE_AUTO_LINKED_WORKTREE,
    ];
    let restore = names
        .into_iter()
        .map(|name| (name, std::env::var_os(name)))
        .collect::<Vec<_>>();
    std::env::set_var(greppy_core::PROJECT_IDENTITY_ENV, &project);
    std::env::set_var(ENV_DISABLE_AUTO_LINKED_WORKTREE, "1");

    let outcome = (|| {
        // Keep an existing worktree pinned to its verified Base. Advancing the
        // primary checkout must not force every already-indexed worktree to
        // build a new repository-wide Base on its next Delta refresh.
        let persisted = overlay_environment(root)?;
        let reused_persisted = match persisted {
            Some((_, commit)) => {
                reuse_verified_base_store(&primary, &commit, shared_data_root, &project)?
                    .map(|prepared| (commit, prepared))
            }
            None => None,
        };
        let (base_commit, prepared) = match reused_persisted {
            Some(binding) => binding,
            None => {
                let base_commit = git_output(&primary, &["rev-parse", "HEAD"])?;
                let prepared = match reuse_verified_base_store(
                    &primary,
                    &base_commit,
                    shared_data_root,
                    &project,
                )? {
                    Some(prepared) => prepared,
                    None => {
                        // Only the first worktree for this immutable Git tree
                        // needs a clean materialization. Every later worktree
                        // opens the hash-verified published Base directly.
                        report_base_phase(progress_path, "preparing_base_checkout");
                        let clean = TemporaryBaseWorktree::create(
                            &primary,
                            shared_data_root,
                            &base_commit,
                        )?;
                        prepare_base_store_paths(
                            &primary,
                            clean.path(),
                            clean.path(),
                            &base_commit,
                            shared_data_root,
                            embedding_args,
                            progress_path,
                        )?
                    }
                };
                (base_commit, prepared)
            }
        };
        configure_overlay_environment(&prepared, &base_commit);
        eprintln!(
            "greppy index: linked worktree uses shared Base {} at {} ({}); only the Git/dirty Delta will be indexed",
            &prepared.identity_hash[..12],
            base_commit,
            if prepared.reused { "reused" } else { "created" },
        );
        Ok::<_, Error>(prepared)
    })();

    match outcome {
        Ok(prepared) => Ok(Some(AutoLinkedWorktreeOverlay {
            _prepared: prepared,
            restore,
        })),
        Err(error) => {
            for (name, value) in restore.into_iter().rev() {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Err(error)
        }
    }
}

struct TemporaryBaseWorktree {
    primary: PathBuf,
    path: PathBuf,
    _parent: tempfile::TempDir,
}

impl TemporaryBaseWorktree {
    fn create(primary: &Path, shared_data_root: &Path, base_commit: &str) -> Result<Self> {
        #[cfg(debug_assertions)]
        if std::env::var_os(ENV_TEST_FORBID_TEMP_BASE_CHECKOUT).is_some() {
            return Err(Error::Invalid(
                "test forbids a second temporary Base checkout".into(),
            ));
        }
        std::fs::create_dir_all(shared_data_root)
            .map_err(|error| Error::io("create shared Base root", error))?;
        let parent = tempfile::Builder::new()
            .prefix("greppy-linked-base-checkout-")
            .tempdir_in(shared_data_root)
            .map_err(|error| Error::io("create clean Base checkout parent", error))?;
        let path = parent.path().join("worktree");
        let output = Command::new("git")
            .arg("-C")
            .arg(primary)
            .args(["worktree", "add", "--detach", "--force"])
            .arg(&path)
            .arg(base_commit)
            .output()
            .map_err(|error| Error::io("create clean Base checkout", error))?;
        if !output.status.success() {
            return Err(Error::Invalid(format!(
                "cannot create clean Base checkout: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(Self {
            primary: primary.to_path_buf(),
            path,
            _parent: parent,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

fn base_identity(workspace: &greppy_agent::workspace::AgentWorkspace) -> Result<BaseStoreIdentity> {
    let repo = workspace.repo_root();
    let canonical_repository_identity = canonical_repository_identity(repo)?;
    let tree_expr = format!("{}^{{tree}}", workspace.base_commit());
    let base_tree_oid = git_output(repo, &["rev-parse", &tree_expr])?;
    let git_object_format = git_output(repo, &["rev-parse", "--show-object-format"])
        .unwrap_or_else(|_| {
            if base_tree_oid.len() == 64 {
                "sha256"
            } else {
                "sha1"
            }
            .into()
        });
    let embedding = crate::embedding_config_for_required_use(crate::EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })?;
    let summary_model = crate::qwen_summary_config_optional()?
        .map(|config| {
            format!(
                "{}#{}",
                crate::qwen_summary_model_key(&config),
                crate::SUMMARY_CACHE_GENERATION
            )
        })
        .unwrap_or_else(|| {
            format!(
                "unavailable/{}#{}",
                greppy_qwen35_native::PROMPT_VERSION,
                crate::SUMMARY_CACHE_GENERATION
            )
        });
    Ok(BaseStoreIdentity {
        format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
        canonical_repository_identity,
        git_object_format,
        base_tree_oid,
        store_schema_version: greppy_store::migrate::CURRENT_VERSION,
        indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
        parser_and_extractor_versions: format!(
            "greppy-parser/extractor-{}",
            env!("CARGO_PKG_VERSION")
        ),
        summary_model_and_prompt_version: summary_model,
        embedding_model: embedding.model_id,
        embedding_prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
        embedding_dimensions: greppy_embed_native::EMBEDDING_DIM,
        embedding_encoding: "f32+i8-v1".into(),
    })
}

/// Reuse an already-published Base without ever building one synchronously.
/// Interactive startup uses this fast path so a cold Base cannot hide a full
/// embedding run behind an opaque pre-TUI phase. (Branch TUI fast path; the
/// verified reuse below is the core campaign's path.)
pub(crate) fn try_reuse_base_store(
    workspace: &greppy_agent::workspace::AgentWorkspace,
    shared_data_root: &Path,
) -> Result<Option<PreparedBase>> {
    let identity = base_identity(workspace)?;
    let layout = BaseStoreLayout::new(shared_data_root, &identity)
        .map_err(|error| Error::io("construct Base Store layout", error))?;
    let Ok(manifest) = layout.read_verified_manifest() else {
        return Ok(None);
    };
    if validate_base_contents(workspace.worktree_path(), &layout.graph, &identity).is_err()
        || validate_base_summary_cache(
            workspace.worktree_path(),
            &layout.graph,
            &layout.summary_cache,
            &identity,
        )
        .is_err()
    {
        return Ok(None);
    }
    prepared_base_with_reader(&layout, manifest, true).map(Some)
}

fn reuse_verified_base_store(
    repo_root: &Path,
    base_commit: &str,
    shared_data_root: &Path,
    project: &str,
) -> Result<Option<PreparedBase>> {
    let identity = base_identity_parts(repo_root, base_commit)?;
    let layout = BaseStoreLayout::new(shared_data_root, &identity)
        .map_err(|error| Error::io("construct Base Store layout", error))?;
    let Ok(manifest) = layout.read_verified_manifest() else {
        return Ok(None);
    };
    if validate_base_contents_for_project(project, &layout.graph, &identity).is_err()
        || greppy_store::SummaryCache::open_read_only(
            layout
                .summary_cache
                .parent()
                .ok_or_else(|| Error::Invalid("Base summary cache has no parent".into()))?,
        )
        .is_err()
    {
        return Ok(None);
    }
    prepared_base_with_reader(&layout, manifest, true).map(Some)
}

impl Drop for TemporaryBaseWorktree {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .arg("-C")
            .arg(&self.primary)
            .args(["worktree", "remove", "--force"])
            .arg(&self.path)
            .status();
    }
}

fn primary_worktree_root(root: &Path) -> Result<PathBuf> {
    let expected_repository = canonical_repository_identity(root)?;
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["worktree", "list", "--porcelain", "-z"])
        .output()
        .map_err(|error| Error::io("list linked Git worktrees", error))?;
    if !output.status.success() {
        return Err(Error::Invalid(format!(
            "cannot list linked Git worktrees: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    for field in output.stdout.split(|byte| *byte == 0) {
        let Some(path) = field.strip_prefix(b"worktree ") else {
            continue;
        };
        let path = std::str::from_utf8(path)
            .map_err(|_| Error::Invalid("Git worktree path is not valid UTF-8".into()))?;
        let candidate = PathBuf::from(path);
        if candidate.join(".git").is_dir()
            && canonical_repository_identity(&candidate)? == expected_repository
        {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    Err(Error::Invalid(format!(
        "linked worktree {} has no available primary checkout; restore the primary checkout before indexing",
        root.display()
    )))
}

pub(crate) fn prepare_base_store(
    workspace: &greppy_agent::workspace::AgentWorkspace,
    shared_data_root: &Path,
    embedding_args: crate::EmbeddingCliArgs<'_>,
) -> Result<PreparedBase> {
    prepare_base_store_paths(
        workspace.repo_root(),
        workspace.repository_path(),
        workspace.worktree_path(),
        workspace.base_commit(),
        shared_data_root,
        embedding_args,
        None,
    )
}

fn prepare_base_store_paths(
    repo_root: &Path,
    source_path: &Path,
    worktree_path: &Path,
    base_commit: &str,
    shared_data_root: &Path,
    embedding_args: crate::EmbeddingCliArgs<'_>,
    progress_path: Option<&Path>,
) -> Result<PreparedBase> {
    let identity = base_identity_parts(repo_root, base_commit)?;
    let identity_hash = identity
        .hash()
        .map_err(|error| Error::io("hash Base Store identity", error))?;
    let layout = BaseStoreLayout::new(shared_data_root, &identity)
        .map_err(|error| Error::io("construct Base Store layout", error))?;
    if let Ok(manifest) = layout.read_verified_manifest() {
        if validate_base_contents(worktree_path, &layout.graph, &identity).is_ok()
            && validate_base_summary_cache(
                worktree_path,
                &layout.graph,
                &layout.summary_cache,
                &identity,
            )
            .is_ok()
        {
            return prepared_base_with_reader(&layout, manifest, true);
        }
    }

    // Never disappear into a blocking flock behind another worktree's Base
    // build. That build can legitimately take minutes, but this caller must
    // remain observable and bounded so agents can retry the completed Base
    // instead of abandoning Greppy as hung.
    let builder_lease =
        acquire_base_builder(&layout, &identity_hash, progress_path, BASE_BUILDER_WAIT)?;
    if let Ok(manifest) = layout.read_verified_manifest() {
        if validate_base_contents(worktree_path, &layout.graph, &identity).is_ok()
            && validate_base_summary_cache(
                worktree_path,
                &layout.graph,
                &layout.summary_cache,
                &identity,
            )
            .is_ok()
        {
            drop(builder_lease);
            return prepared_base_with_reader(&layout, manifest, true);
        }
    }
    layout
        .quarantine_current()
        .map_err(|error| Error::io("quarantine invalid Base Store", error))?;
    report_base_phase(progress_path, "validating_base_inventory");
    let expected_file_count = validate_workspace_inventory(source_path, worktree_path)?;

    std::fs::create_dir_all(shared_data_root)
        .map_err(|error| Error::io("create shared Base data root", error))?;
    let staging = tempfile::Builder::new()
        .prefix("greppy-base-build-")
        .tempdir_in(shared_data_root)
        .map_err(|error| Error::io("create Base build staging directory", error))?;
    let staging_data = staging.path().join("data");
    std::fs::create_dir_all(&staging_data)
        .map_err(|error| Error::io("create Base build data directory", error))?;
    let binary = std::env::current_exe()
        .map_err(|error| Error::io("resolve current greppy binary for Base build", error))?;
    report_base_phase(progress_path, "building_base_graph");
    let mut command = Command::new(binary);
    command.arg("index");
    #[cfg(any(
        feature = "ci-test-assets",
        debug_assertions,
        feature = "store-cow-release-perf"
    ))]
    if crate::test_inference_skipped() {
        command.env(crate::ENV_TEST_FORCE_EMBED_COMPLETION, "1");
    }
    append_embedding_cli_args(&mut command, embedding_args);
    command
        .current_dir(worktree_path)
        .env("GREPPY_STORE_DIR", &staging_data)
        // A published Base is not valid until every candidate has its vector;
        // never let the ordinary foreground-index lazy threshold hand this
        // build to a background process outside the publication lease.
        .env("GREPPY_LAZY_EMBED_MIN_SPANS", usize::MAX.to_string())
        .env(ENV_DISABLE_AUTO_LINKED_WORKTREE, "1")
        .env_remove("GREPPY_BACKGROUND_JOB")
        .env_remove("GREPPY_BACKGROUND_CAUSE")
        .env_remove("GREPPY_BACKGROUND_KIND")
        .env_remove("GREPPY_BACKGROUND_STARTED_AT")
        .env_remove("GREPPY_BACKGROUND_TARGET_GENERATION")
        .env_remove(ENV_MODE)
        .env_remove(ENV_BASE_PATH)
        .env_remove(ENV_BASE_COMMIT)
        .stdin(Stdio::null())
        .stdout(Stdio::null());
    if let Some(path) = progress_path {
        command.env(crate::ENV_DELEGATED_BACKGROUND_JOB, path);
    }
    let status = command
        .status()
        .map_err(|error| Error::io("start immutable Base index build", error))?;
    if !status.success() {
        return Err(Error::Invalid(format!(
            "immutable Base index build exited {status}"
        )));
    }
    let staged_graph = staging_data
        .join("workspaces")
        .join(format!("v{}", greppy_core::cache::STORE_FORMAT_VERSION))
        .join(greppy_core::workspace_hash(worktree_path))
        .join("graph.db");
    if !staged_graph.is_file() {
        return Err(Error::Invalid(format!(
            "Base build succeeded without graph.db at {}",
            staged_graph.display()
        )));
    }
    {
        let store = greppy_store::Store::open_with(
            &staged_graph,
            greppy_store::OpenOptions::query_writer(),
        )?;
        store
            .conn()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .map_err(|error| Error::Store(format!("checkpoint Base graph: {error}")))?;
    }
    validate_base_contents(worktree_path, &staged_graph, &identity)?;
    validate_base_file_count(&staged_graph, expected_file_count)?;
    #[cfg(debug_assertions)]
    if std::env::var_os(ENV_TEST_BASE_SUMMARY_FAIL).is_some() {
        return Err(Error::Invalid(
            "injected immutable Base summary cache publication failure".into(),
        ));
    }
    report_base_phase(progress_path, "initializing_base_summary_cache");
    let staged_summary_cache =
        build_base_summary_cache(&staged_graph, &identity.summary_model_and_prompt_version)?;
    validate_base_summary_cache(
        worktree_path,
        &staged_graph,
        &staged_summary_cache,
        &identity,
    )?;
    let manifest = layout
        .publish_graph_with_summary(identity, &staged_graph, &staged_summary_cache)
        .map_err(|error| Error::io("publish immutable Base Store", error))?;
    drop(builder_lease);
    prepared_base_with_reader(&layout, manifest, false)
}

fn validate_workspace_inventory(source_path: &Path, worktree_path: &Path) -> Result<usize> {
    fn inventory(root: &Path) -> Result<Vec<(String, Option<u64>)>> {
        greppy_discover::walk(root)
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|entry| (entry.rel_path, entry.size))
                    .collect()
            })
            .map_err(|error| {
                Error::Invalid(format!(
                    "cannot inventory immutable Base candidate {}: {error}",
                    root.display()
                ))
            })
    }

    let expected = inventory(source_path)?;
    let actual = inventory(worktree_path)?;
    if expected == actual {
        return Ok(expected.len());
    }
    let first_difference = expected
        .iter()
        .zip(actual.iter())
        .find(|(left, right)| left != right)
        .map(|(left, right)| format!("expected {left:?}, mounted {right:?}"))
        .or_else(|| {
            expected
                .get(actual.len())
                .map(|entry| format!("missing mounted entry {entry:?}"))
        })
        .or_else(|| {
            actual
                .get(expected.len())
                .map(|entry| format!("unexpected mounted entry {entry:?}"))
        })
        .unwrap_or_else(|| "inventory content differs".into());
    Err(Error::Invalid(format!(
        "portable workspace inventory is incomplete: source has {} files, mount has {}; {first_difference}",
        expected.len(),
        actual.len()
    )))
}

fn validate_base_file_count(graph_path: &Path, expected: usize) -> Result<()> {
    let store = greppy_store::Store::open_with(graph_path, greppy_store::OpenOptions::read_only())?;
    let actual = store
        .conn()
        .query_row("SELECT COUNT(*) FROM file_state", [], |row| {
            row.get::<_, usize>(0)
        })
        .map_err(|error| Error::Store(format!("count Base file inventory: {error}")))?;
    if actual != expected {
        return Err(Error::Invalid(format!(
            "Base file inventory is incomplete: expected {expected} file_state rows, found {actual}"
        )));
    }
    Ok(())
}

fn append_embedding_cli_args(command: &mut Command, embedding_args: crate::EmbeddingCliArgs<'_>) {
    if let Some(device) = embedding_args.device {
        command.arg("--device").arg(device);
    }
    if embedding_args.no_gpu {
        command.arg("--no-gpu");
    }
}

fn validate_base_contents(
    root: &Path,
    graph_path: &Path,
    identity: &BaseStoreIdentity,
) -> Result<()> {
    validate_base_contents_for_project(&greppy_core::project_identity(root), graph_path, identity)
}

fn validate_base_contents_for_project(
    project: &str,
    graph_path: &Path,
    identity: &BaseStoreIdentity,
) -> Result<()> {
    let store = greppy_store::Store::open_with(graph_path, greppy_store::OpenOptions::read_only())?;
    store.integrity_check()?;
    let schema_version: Option<u32> = store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|value| value.parse().ok());
    if schema_version != Some(identity.store_schema_version) {
        return Err(Error::Invalid(format!(
            "Base schema is incompatible: expected {}, got {}",
            identity.store_schema_version,
            schema_version
                .map(|value| value.to_string())
                .unwrap_or_else(|| "missing".into())
        )));
    }
    let generation = store
        .list_workspace_states()?
        .into_iter()
        .map(|state| state.graph_generation)
        .max()
        .ok_or_else(|| Error::Invalid("Base build has no workspace generation".into()))?;
    let completion_key = crate::embedding_complete_key(project);
    let completion: Option<String> = store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [&completion_key],
            |row| row.get(0),
        )
        .ok();
    let expected_completion = format!("{generation}|{}", identity.embedding_model);
    #[cfg(debug_assertions)]
    let injected_summary_failure = std::env::var_os(ENV_TEST_BASE_SUMMARY_FAIL).is_some();
    #[cfg(not(debug_assertions))]
    let injected_summary_failure = false;
    if !injected_summary_failure && completion.as_deref() != Some(expected_completion.as_str()) {
        return Err(Error::Invalid(format!(
            "Base embedding generation is incomplete: expected `{expected_completion}`, got {}",
            completion.as_deref().unwrap_or("missing")
        )));
    }
    let provider_failures = store
        .list_provider_states(project)?
        .into_iter()
        .filter(|provider| provider.status != "unsupported")
        .map(|provider| provider.files_failed.max(0) as u64)
        .sum::<u64>();
    if provider_failures > 0 {
        return Err(Error::Invalid(format!(
            "Base build has {provider_failures} provider file failures"
        )));
    }
    Ok(())
}

fn prepared_base_with_reader(
    layout: &BaseStoreLayout,
    manifest: greppy_store::BaseStoreManifest,
    reused: bool,
) -> Result<PreparedBase> {
    let reader_lease = layout
        .acquire_reader(false)
        .map_err(|error| Error::io("acquire Base Store reader lease", error))?
        .ok_or_else(|| Error::Lock("blocking Base reader lease returned no guard".into()))?;
    greppy_core::cache::touch_last_used_dir(&layout.directory);
    Ok(PreparedBase {
        graph_path: layout.graph.clone(),
        identity_hash: manifest.identity_hash,
        reused,
        _reader_lease: reader_lease,
    })
}

fn build_base_summary_cache(graph_path: &Path, expected_model_key: &str) -> Result<PathBuf> {
    let summary_dir = graph_path
        .parent()
        .ok_or_else(|| Error::Invalid("staged Base graph has no parent".into()))?
        .join("base-summary-cache");
    let cache = greppy_store::SummaryCache::open(&summary_dir)?;
    let Some(config) = crate::qwen_summary_config_optional()? else {
        let unavailable_key = format!(
            "unavailable/{}#{}",
            greppy_qwen35_native::PROMPT_VERSION,
            crate::SUMMARY_CACHE_GENERATION
        );
        if unavailable_key != expected_model_key {
            return Err(Error::Invalid(
                "Base summary model identity changed during publication".into(),
            ));
        }
        if cache.count()? != 0 {
            return Err(Error::Invalid(
                "summary-disabled Base cache unexpectedly contains entries".into(),
            ));
        }
        drop(cache);
        return Ok(summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE));
    };
    let model_key = crate::qwen_summary_model_key(&config);
    let complete_model_key = format!("{model_key}#{}", crate::SUMMARY_CACHE_GENERATION);
    if complete_model_key != expected_model_key {
        return Err(Error::Invalid(
            "Base summary model identity changed during publication".into(),
        ));
    }
    // Summaries are derived navigation output, not graph correctness data.
    // Eagerly generating one summary for every definition made a cold Base
    // take hours and allowed a transient summary daemon failure to discard an
    // otherwise complete graph and embedding generation. Publish a verified
    // empty cache bound to the model identity; navigation fills the private
    // workspace cache lazily on an actual summary request.
    if cache.count()? != 0 {
        return Err(Error::Invalid(
            "new Base summary cache unexpectedly contains entries".into(),
        ));
    }
    drop(cache);
    Ok(summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE))
}

fn expected_base_summary_spans(
    root: &Path,
    graph_path: &Path,
) -> Result<Vec<(String, i64, String, String)>> {
    let store = greppy_store::Store::open_with(graph_path, greppy_store::OpenOptions::read_only())?;
    let project = greppy_core::project_identity(root);
    let mut expected = std::collections::BTreeMap::new();
    for node in store.list_nodes(&project, "", "", 0, usize::MAX)? {
        if node.file_path.is_empty() || node.start_line <= 0 || node.end_line < node.start_line {
            continue;
        }
        let Some(span) = crate::read_span_with_meta(
            root,
            &node.file_path,
            node.start_line,
            node.end_line,
            crate::CONTEXT_SPAN_CAP,
            false,
        ) else {
            continue;
        };
        if span.text.trim().is_empty() {
            continue;
        }
        let semantic_span = crate::read_span_with_meta(
            root,
            &node.file_path,
            node.start_line,
            node.end_line,
            crate::SEMANTIC_PURPOSE_SPAN_CAP_LINES,
            false,
        )
        .map(|span| crate::cap_semantic_purpose_span(&span.text));
        for source in std::iter::once(span.text.as_str())
            .chain(semantic_span.as_deref())
            .filter(|source| !source.trim().is_empty())
        {
            let hash = greppy_store::span_hash(&node.file_path, source);
            expected.entry(hash.clone()).or_insert_with(|| {
                (
                    node.file_path.clone(),
                    node.start_line,
                    source.to_string(),
                    hash,
                )
            });
        }
    }
    Ok(expected.into_values().collect())
}

fn validate_base_summary_cache(
    root: &Path,
    graph_path: &Path,
    summary_path: &Path,
    identity: &BaseStoreIdentity,
) -> Result<()> {
    let directory = summary_path
        .parent()
        .ok_or_else(|| Error::Invalid("Base summary cache has no parent".into()))?;
    let cache = greppy_store::SummaryCache::open_read_only(directory)?;
    let actual = cache.count()? as usize;
    if identity
        .summary_model_and_prompt_version
        .starts_with("unavailable/")
    {
        if actual != 0 {
            return Err(Error::Invalid(
                "summary-disabled Base cache unexpectedly contains entries".into(),
            ));
        }
        return Ok(());
    }
    // An empty Base summary cache is the intentional lazy-summary contract.
    // The manifest still authenticates the empty SQLite file and the Base
    // identity still pins the model/prompt generation. Non-empty legacy or
    // externally warmed caches are validated below for bounded completeness.
    if actual == 0 {
        return Ok(());
    }
    let expected = expected_base_summary_spans(root, graph_path)?;
    if actual > expected.len() {
        return Err(Error::Invalid(format!(
            "Base summary cache has more entries than the visible graph: maximum {}, found {actual}",
            expected.len()
        )));
    }
    for (_, _, _, hash) in expected {
        if cache
            .get(&identity.summary_model_and_prompt_version, &hash)?
            .is_none()
        {
            return Err(Error::Invalid(format!(
                "Base summary cache is missing span {hash}"
            )));
        }
    }
    Ok(())
}

fn base_identity_parts(repo: &Path, base_commit: &str) -> Result<BaseStoreIdentity> {
    let canonical_repository_identity = canonical_repository_identity(repo)?;
    let tree_expr = format!("{base_commit}^{{tree}}");
    let base_tree_oid = git_output(repo, &["rev-parse", &tree_expr])?;
    let git_object_format = git_output(repo, &["rev-parse", "--show-object-format"])
        .unwrap_or_else(|_| {
            if base_tree_oid.len() == 64 {
                "sha256"
            } else {
                "sha1"
            }
            .into()
        });
    let embedding = crate::embedding_config_for_required_use(crate::EmbeddingCliArgs {
        device: None,
        no_gpu: false,
    })?;
    let summary_model = crate::qwen_summary_config_optional()?
        .map(|config| {
            format!(
                "{}#{}",
                crate::qwen_summary_model_key(&config),
                crate::SUMMARY_CACHE_GENERATION
            )
        })
        .unwrap_or_else(|| {
            format!(
                "unavailable/{}#{}",
                greppy_qwen35_native::PROMPT_VERSION,
                crate::SUMMARY_CACHE_GENERATION
            )
        });
    Ok(BaseStoreIdentity {
        format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
        canonical_repository_identity,
        git_object_format,
        base_tree_oid,
        store_schema_version: greppy_store::migrate::CURRENT_VERSION,
        indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
        parser_and_extractor_versions: format!(
            "greppy-parser/extractor-{}",
            env!("CARGO_PKG_VERSION")
        ),
        summary_model_and_prompt_version: summary_model,
        embedding_model: embedding.model_id,
        embedding_prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
        embedding_dimensions: greppy_embed_native::EMBEDDING_DIM,
        embedding_encoding: "f32+i8-v1".into(),
    })
}

pub(crate) fn canonical_repository_identity(repo: &Path) -> Result<String> {
    let common_dir = git_output(
        repo,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )
    .or_else(|_| git_output(repo, &["rev-parse", "--git-common-dir"]))?;
    let common_path = {
        let path = PathBuf::from(&common_dir);
        let absolute = if path.is_absolute() {
            path
        } else {
            repo.join(path)
        };
        absolute.canonicalize().unwrap_or(absolute)
    };
    Ok(format!("git-common-dir:{}", common_path.display()))
}

pub(crate) fn visibility_against(root: &Path, base_commit: &str) -> Result<VisibilityIndex> {
    let diff = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            base_commit,
            "--",
        ])
        .output()
        .map_err(|error| Error::io("run git diff for Store Delta", error))?;
    if !diff.status.success() {
        return Err(Error::Invalid(format!(
            "git diff against pinned Base {base_commit} failed: {}",
            String::from_utf8_lossy(&diff.stderr).trim()
        )));
    }
    let fields = nul_fields(&diff.stdout)?;
    let mut dirty = Vec::new();
    let mut deleted = Vec::new();
    let mut index = 0;
    while index < fields.len() {
        let status = fields[index].as_str();
        index += 1;
        let kind = status.as_bytes().first().copied().unwrap_or_default();
        match kind {
            b'R' => {
                let old = take_field(&fields, &mut index, status)?;
                let new = take_field(&fields, &mut index, status)?;
                deleted.push(old);
                dirty.push(new);
            }
            b'C' => {
                let _old = take_field(&fields, &mut index, status)?;
                dirty.push(take_field(&fields, &mut index, status)?);
            }
            b'D' => deleted.push(take_field(&fields, &mut index, status)?),
            b'A' | b'M' | b'T' | b'U' => dirty.push(take_field(&fields, &mut index, status)?),
            _ => {
                return Err(Error::Invalid(format!(
                    "unsupported git diff status `{status}` for Store Delta"
                )))
            }
        }
    }

    let untracked = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--others", "--exclude-standard", "-z"])
        .output()
        .map_err(|error| Error::io("list untracked files for Store Delta", error))?;
    if !untracked.status.success() {
        return Err(Error::Invalid(format!(
            "git ls-files for Store Delta failed: {}",
            String::from_utf8_lossy(&untracked.stderr).trim()
        )));
    }
    dirty.extend(nul_fields(&untracked.stdout)?);
    VisibilityIndex::new(dirty, deleted)
        .map_err(|error| Error::io("validate Store Delta visibility", error))
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| Error::io(format!("run git {}", args.join(" ")), error))?;
    if !output.status.success() {
        return Err(Error::Invalid(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|_| Error::Invalid(format!("git {} returned non-UTF-8", args.join(" "))))?;
    let value = value.trim().to_string();
    if value.is_empty() {
        return Err(Error::Invalid(format!(
            "git {} returned empty output",
            args.join(" ")
        )));
    }
    Ok(value)
}

fn nul_fields(bytes: &[u8]) -> Result<Vec<String>> {
    bytes
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .map(|field| {
            String::from_utf8(field.to_vec())
                .map_err(|_| Error::Invalid("Store Delta path is not valid UTF-8".into()))
        })
        .collect()
}

fn take_field(fields: &[String], index: &mut usize, status: &str) -> Result<String> {
    let value = fields.get(*index).cloned().ok_or_else(|| {
        Error::Invalid(format!("truncated git diff record after status `{status}`"))
    })?;
    *index += 1;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn fixture() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "cow@test.invalid"]);
        git(tmp.path(), &["config", "user.name", "Store CoW"]);
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(tmp.path().join("src/b.rs"), "fn b() {}\n").unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "base"]);
        tmp
    }

    #[test]
    fn private_delta_paths_exclude_structural_folders() {
        let mut store = greppy_store::Store::open_memory().unwrap();
        store
            .upsert_project(&greppy_store::Project {
                name: "p".into(),
                indexed_at: "2026-09-01T00:00:00Z".into(),
                root_path: "/repo".into(),
            })
            .unwrap();
        for (label, name, path) in [("Folder", "src", "src"), ("Function", "run", "src/lib.rs")] {
            store
                .insert_node(&greppy_store::NewNode {
                    project: "p".into(),
                    label: label.into(),
                    name: name.into(),
                    qualified_name: format!("p::{name}"),
                    file_path: path.into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
        }
        let paths = private_delta_paths(&store).unwrap();
        assert!(!paths.contains("src"));
        assert!(paths.contains("src/lib.rs"));
    }

    #[test]
    fn concurrent_base_builder_wait_is_bounded_and_actionable() {
        let repo = fixture();
        let commit = git(repo.path(), &["rev-parse", "HEAD"]);
        let identity = base_identity_parts(repo.path(), &commit).unwrap();
        let identity_hash = identity.hash().unwrap();
        let data_root = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(data_root.path(), &identity).unwrap();
        let _held = layout.acquire_builder(true).unwrap().unwrap();
        let progress_path = data_root.path().join("index.job");
        crate::write_background_job(
            &progress_path,
            &serde_json::json!({
                "schema_version": crate::BACKGROUND_JOB_SCHEMA_VERSION,
                "kind": "index",
                "pid": std::process::id(),
                "started_at_unix_secs": 1,
                "updated_at_unix_secs": 1,
                "state": "preparing_base_checkout"
            }),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let error = match acquire_base_builder(
            &layout,
            &identity_hash,
            Some(&progress_path),
            std::time::Duration::from_millis(20),
        ) {
            Ok(_) => panic!("second Base builder unexpectedly acquired the held lease"),
            Err(error) => error,
        };

        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        let message = error.to_string();
        assert!(message.contains("another worktree is building immutable Base"));
        assert!(message.contains(&identity_hash));
        assert!(message.contains("rerun `greppy index`"));
        assert!(message.contains(
            layout
                .builder_lock_path()
                .unwrap()
                .to_string_lossy()
                .as_ref()
        ));
        let progress = crate::read_background_job(&progress_path).unwrap();
        assert_eq!(progress["state"], "waiting_for_base_builder");
        assert_eq!(progress["progress_unit"], "steps");
        assert_eq!(progress["completed_spans"], 0);
        assert_eq!(progress["total_spans"], 0);
    }

    #[test]
    fn immutable_base_build_preserves_explicit_embedding_device_contract() {
        let mut cuda = Command::new("greppy");
        cuda.arg("index");
        append_embedding_cli_args(
            &mut cuda,
            crate::EmbeddingCliArgs {
                device: Some("cuda"),
                no_gpu: false,
            },
        );
        assert_eq!(
            cuda.get_args().collect::<Vec<_>>(),
            ["index", "--device", "cuda"]
        );

        let mut cpu_only = Command::new("greppy");
        cpu_only.arg("index");
        append_embedding_cli_args(
            &mut cpu_only,
            crate::EmbeddingCliArgs {
                device: None,
                no_gpu: true,
            },
        );
        assert_eq!(
            cpu_only.get_args().collect::<Vec<_>>(),
            ["index", "--no-gpu"]
        );
    }

    #[test]
    fn visibility_is_pinned_to_base_and_handles_revert_delete_and_untracked() {
        let repo = fixture();
        let base = git(repo.path(), &["rev-parse", "HEAD"]);
        std::fs::write(repo.path().join("src/a.rs"), "fn changed() {}\n").unwrap();
        std::fs::remove_file(repo.path().join("src/b.rs")).unwrap();
        std::fs::write(repo.path().join("src/new.rs"), "fn new() {}\n").unwrap();
        let visibility = visibility_against(repo.path(), &base).unwrap();
        assert!(visibility.is_dirty_path("src/a.rs"));
        assert!(visibility.is_dirty_path("src/new.rs"));
        assert!(visibility.is_deleted_path("src/b.rs"));

        std::fs::write(repo.path().join("src/a.rs"), "fn a() {}\n").unwrap();
        let reverted = visibility_against(repo.path(), &base).unwrap();
        assert!(!reverted.hides_base_path("src/a.rs"));
        assert_eq!(reverted.changed_count(), 2);
    }

    #[test]
    fn visibility_represents_rename_as_delete_plus_add() {
        let repo = fixture();
        let base = git(repo.path(), &["rev-parse", "HEAD"]);
        std::fs::rename(
            repo.path().join("src/a.rs"),
            repo.path().join("src/renamed.rs"),
        )
        .unwrap();
        let visibility = visibility_against(repo.path(), &base).unwrap();
        assert!(visibility.is_deleted_path("src/a.rs"));
        assert!(visibility.is_dirty_path("src/renamed.rs"));
    }

    #[test]
    fn discovery_filtered_delta_freshness_falls_back_to_content_hash() {
        let repo = fixture();
        let hidden = repo.path().join(".github/workflows/ci.yml");
        std::fs::create_dir_all(hidden.parent().unwrap()).unwrap();
        std::fs::write(&hidden, "name: CI\n").unwrap();

        let mut store = greppy_store::Store::open_memory().unwrap();
        let options = greppy_indexer::IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                ".github/workflows/ci.yml".to_string(),
            ])),
            ..greppy_indexer::IndexOptions::default()
        };
        greppy_indexer::index_with_options(&mut store, repo.path(), "p", &options).unwrap();
        let skip = store
            .get_index_skip("p", ".github/workflows/ci.yml")
            .unwrap()
            .expect("hidden path skip identity");
        assert_eq!(skip.reason, "discovery_filtered");

        // An absent/mismatched stat identity forces the content fallback.
        // The unchanged bytes must still prove the Delta snapshot fresh.
        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            ".github/workflows/ci.yml",
            &std::collections::HashMap::new(),
        )
        .unwrap());
    }

    #[test]
    fn visibility_diagnostics_report_only_the_actual_manifest_delta() {
        let cached = VisibilityIndex::new(
            ["kept.rs".into(), "removed.rs".into()],
            ["was-deleted.rs".into()],
        )
        .unwrap();
        let live = VisibilityIndex::new(
            ["kept.rs".into(), "was-deleted.rs".into()],
            ["now-deleted.rs".into()],
        )
        .unwrap();
        assert_eq!(
            visibility_changed_paths(&cached, &live),
            ["now-deleted.rs", "removed.rs", "was-deleted.rs"]
        );
    }
}
