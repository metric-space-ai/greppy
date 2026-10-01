//! Store-CoW lifecycle shared by `greppy -p`, index warming, and query opens.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use greppy_core::error::{Error, Result};
use greppy_store::{BaseBuilderLease, BaseStoreIdentity, BaseStoreLayout, VisibilityIndex};
use rusqlite::OptionalExtension;
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
const RUST_CALLER_EDGES_REPAIR_META_KEY: &str = greppy_indexer::RUST_CALLER_EDGES_REPAIR_META_KEY;
const RUST_CALLER_EDGES_REPAIR_COMPLETE: &str = greppy_indexer::RUST_CALLER_EDGES_REPAIR_COMPLETE;
const BASE_EMBEDDING_DEFERRED_META_PREFIX: &str = "store_cow.embedding_deferred.v1:";
#[cfg(debug_assertions)]
const ENV_TEST_BASE_SUMMARY_FAIL: &str = "GREPPY_TEST_BASE_SUMMARY_FAIL";
#[cfg(debug_assertions)]
const ENV_TEST_FORBID_TEMP_BASE_CHECKOUT: &str = "GREPPY_TEST_FORBID_TEMP_BASE_CHECKOUT";

#[derive(Debug, Clone)]
pub(crate) struct OverlaySpec {
    pub base_path: PathBuf,
    pub base_commit: String,
    pub visibility: VisibilityIndex,
}

#[derive(Debug, Clone)]
struct PersistedOverlayBinding {
    base_path: PathBuf,
    base_commit: String,
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
        base_commit,
        visibility,
    }))
}

pub(crate) fn overlay_environment(root: &Path) -> Result<Option<(PathBuf, String)>> {
    overlay_environment_inner(root, false)
}

/// Read a persisted Delta binding while permitting its Base file to be
/// absent. Recovery may inspect this binding before rebuilding; query readers
/// must wait for that publication and then use the strict overlay open, never
/// attach an incomplete overlay.
pub(crate) fn overlay_environment_for_recovery(root: &Path) -> Result<Option<(PathBuf, String)>> {
    overlay_environment_inner(root, true)
}

fn overlay_environment_inner(
    root: &Path,
    allow_missing_persisted_base: bool,
) -> Result<Option<(PathBuf, String)>> {
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

    let Some(binding) = persisted_overlay_binding(root)? else {
        return Ok(None);
    };
    if !binding.base_path.is_file() && !allow_missing_persisted_base {
        return Err(Error::Invalid(format!(
            "linked-worktree Base Store is missing: {}; run `greppy index` to rebuild it",
            binding.base_path.display()
        )));
    }
    Ok(Some((binding.base_path, binding.base_commit)))
}

fn persisted_overlay_binding(root: &Path) -> Result<Option<PersistedOverlayBinding>> {
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
    Ok(Some(PersistedOverlayBinding {
        base_path,
        base_commit,
    }))
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
    let mut missing_dirty = std::collections::BTreeSet::new();
    for rel_path in &dirty {
        match std::fs::symlink_metadata(root.join(rel_path)) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                missing_dirty.insert((*rel_path).to_owned());
            }
            Err(error) => {
                return Err(Error::io(
                    format!("stat Store-CoW Delta path {rel_path}"),
                    error,
                ));
            }
        }
    }
    let sparse_blobs = persisted_sparse_delta_blobs(root, &missing_dirty)?;
    for rel_path in &dirty {
        if !persisted_delta_path_matches(
            root,
            store,
            project,
            rel_path,
            &identities,
            &sparse_blobs,
        )? {
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
    sparse_blobs: &std::collections::HashMap<String, SparseDeltaBlob>,
) -> Result<bool> {
    let path = root.join(rel_path);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let Some(blob) = sparse_blobs.get(rel_path) else {
                return Ok(false);
            };
            if let Some(state) = store.get_file_state(project, rel_path).map_err(|error| {
                Error::Store(format!("read sparse Store-CoW file state: {error}"))
            })? {
                return Ok(state.size >= 0
                    && blob.size == state.size as u64
                    && blob.sha256 == state.sha256);
            }
            let skip = store
                .get_index_skip(project, rel_path)
                .map_err(|error| Error::Store(format!("read sparse Store-CoW skip: {error}")))?;
            return Ok(skip.is_some_and(|skip| skip.reason == "discovery_filtered"));
        }
        Err(error) => {
            return Err(Error::io(
                format!("stat Store-CoW Delta path {rel_path}"),
                error,
            ));
        }
    };
    let current = greppy_discover::stable_metadata(&metadata);
    if let Some(state) = store
        .get_file_state(project, rel_path)
        .map_err(|error| Error::Store(format!("read Store-CoW file state: {error}")))?
    {
        if !metadata.is_file() {
            return Ok(false);
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparseDeltaBlob {
    size: u64,
    sha256: String,
}

struct ReapedChild(Option<std::process::Child>);

impl ReapedChild {
    fn child_mut(&mut self) -> &mut std::process::Child {
        self.0.as_mut().expect("child is present until wait")
    }

    fn wait(mut self) -> std::io::Result<std::process::ExitStatus> {
        self.0.take().expect("child is present until wait").wait()
    }
}

impl Drop for ReapedChild {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn persisted_sparse_delta_blobs(
    root: &Path,
    rel_paths: &std::collections::BTreeSet<String>,
) -> Result<std::collections::HashMap<String, SparseDeltaBlob>> {
    if rel_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .env("GIT_LITERAL_PATHSPECS", "1")
        .args(["ls-files", "-v", "--stage", "-z", "--"])
        .args(rel_paths);
    let listed = command
        .output()
        .map_err(|error| Error::io("inspect sparse Store-CoW Delta path", error))?;
    if !listed.status.success() {
        return Err(Error::Invalid(format!(
            "git ls-files for sparse Store-CoW Delta failed: {}",
            String::from_utf8_lossy(&listed.stderr).trim()
        )));
    }
    let mut entries = Vec::new();
    for field in nul_fields(&listed.stdout)? {
        let Some((header, rel_path)) = field.split_once('\t') else {
            return Err(Error::Invalid(
                "malformed sparse git ls-files record".into(),
            ));
        };
        let columns = header.split_whitespace().collect::<Vec<_>>();
        if columns.len() != 4 || columns[0] != "S" || columns[3] != "0" {
            continue;
        }
        if !rel_paths.contains(rel_path) {
            return Err(Error::Invalid(format!(
                "git ls-files returned unexpected sparse path `{rel_path}`"
            )));
        }
        entries.push((rel_path.to_owned(), columns[2].to_owned()));
    }
    if entries.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    let child = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| Error::io("read sparse Store-CoW Delta blob", error))?;
    let mut child = ReapedChild(Some(child));
    use std::io::{BufRead, Read, Write};
    let mut stdin = child
        .child_mut()
        .stdin
        .take()
        .ok_or_else(|| Error::Invalid("git cat-file stdin is unavailable".into()))?;
    let stdout = child
        .child_mut()
        .stdout
        .take()
        .ok_or_else(|| Error::Invalid("git cat-file stdout is unavailable".into()))?;
    let mut stdout = std::io::BufReader::new(stdout);
    let mut blobs = std::collections::HashMap::new();
    for (rel_path, expected_oid) in &entries {
        writeln!(&mut stdin, "{expected_oid}")
            .and_then(|_| stdin.flush())
            .map_err(|error| Error::io("request sparse Store-CoW Delta blob", error))?;
        let mut header = String::new();
        stdout
            .read_line(&mut header)
            .map_err(|error| Error::io("read sparse git cat-file header", error))?;
        let header = header.trim_end_matches('\n');
        let columns = header.split_whitespace().collect::<Vec<_>>();
        if columns.len() != 3 || columns[0] != expected_oid || columns[1] != "blob" {
            return Err(Error::Invalid(format!(
                "unexpected sparse git cat-file header `{header}`"
            )));
        }
        let size = columns[2]
            .parse::<usize>()
            .map_err(|_| Error::Invalid(format!("invalid sparse blob size in `{header}`")))?;
        if size as u64 > greppy_freshness::incremental::MAX_FILE_SIZE_BYTES {
            return Err(Error::Invalid(format!(
                "sparse Store-CoW Delta blob `{rel_path}` exceeds the indexed file size limit"
            )));
        }
        let mut content = vec![0; size];
        stdout
            .read_exact(&mut content)
            .map_err(|error| Error::io("read sparse git cat-file content", error))?;
        let mut newline = [0u8; 1];
        stdout
            .read_exact(&mut newline)
            .map_err(|error| Error::io("finish sparse git cat-file content", error))?;
        if newline != [b'\n'] {
            return Err(Error::Invalid(
                "malformed sparse git cat-file content terminator".into(),
            ));
        }
        if blobs
            .insert(
                rel_path.clone(),
                SparseDeltaBlob {
                    size: size as u64,
                    sha256: greppy_store::file_state::sha256_hex(&content),
                },
            )
            .is_some()
        {
            return Err(Error::Invalid(format!(
                "duplicate sparse Store-CoW Delta path `{rel_path}`"
            )));
        }
    }
    drop(stdin);
    let status = child
        .wait()
        .map_err(|error| Error::io("finish sparse Store-CoW Delta blob batch", error))?;
    if !status.success() {
        return Err(Error::Invalid(format!(
            "git cat-file for sparse Store-CoW Delta failed with {status}"
        )));
    }
    Ok(blobs)
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

fn rust_repair_requires_source_refresh(
    store: &greppy_store::Store,
    root: &Path,
    project: &str,
) -> bool {
    let root = root.to_string_lossy();
    crate::freshness::freshness_is_reindexable_stale(&crate::nav_freshness_json_uncached(
        store,
        Some(root.as_ref()),
        project,
    ))
}

/// Repair a Delta from an older Rust path resolver whose workspace state already advertises v7 but
/// whose resolved Rust caller edges were produced by the old resolver.
///
/// The repair consumes the composed visible raw-edge view, including raw edges
/// retained in an immutable Base, and replaces the logical overlay edges with
/// results from the current resolver. Nodes, file state, and vector embeddings
/// remain untouched. The schema-meta marker makes the operation one-shot for
/// an otherwise unchanged Delta; a failed resolution leaves the marker absent
/// so the next query can retry safely.
pub(crate) fn repair_persisted_v7_delta(
    delta_path: &Path,
    base_path: &Path,
    visibility: &VisibilityIndex,
    root: &Path,
    project: &str,
) -> Result<bool> {
    let delta = greppy_store::Store::open_with(delta_path, greppy_store::OpenOptions::read_only())?;
    let pending = persisted_v7_delta_needs_repair(&delta, root)?;
    drop(delta);
    if !pending {
        return Ok(false);
    }

    // A concurrent normal query may have observed the same unmarked Delta.
    // Poll the existing OS lock rather than serving its stale read-only
    // snapshot. Once the writer releases the lock, re-read the marker before
    // electing a repairer; only a missing marker permits resolver work.
    let deadline = std::time::Instant::now() + crate::NAV_FRESHNESS_BUDGET;
    let _lock = loop {
        match greppy_freshness::try_acquire(delta_path) {
            Ok(lock) => break lock,
            Err(greppy_freshness::LockError::Held { path }) => {
                let observed = greppy_store::Store::open_with(
                    delta_path,
                    greppy_store::OpenOptions::read_only(),
                )?;
                if !persisted_v7_delta_needs_repair(&observed, root)? {
                    return Ok(false);
                }
                if std::time::Instant::now() >= deadline {
                    return Err(Error::Lock(format!(
                        "timed out waiting for persisted Delta repair publication; lock {}",
                        path.display()
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(error) => return Err(error.into()),
        }
    };
    let current =
        greppy_store::Store::open_with(delta_path, greppy_store::OpenOptions::query_writer())?;
    if !persisted_v7_delta_needs_repair(&current, root)? {
        return Ok(false);
    }
    drop(current);
    let mut overlay = greppy_store::Store::open_overlay(base_path, delta_path, visibility)?;
    // Defer migration until the normal freshness refresh publishes edited sources.
    // Keep the repair marker pending and never authorize a stale graph.
    if rust_repair_requires_source_refresh(&overlay, root, project) {
        return Ok(false);
    }
    let raw_edges = overlay.list_raw_edges(project)?;
    if raw_edges.is_empty() {
        let existing_edges: i64 = overlay
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM main.overlay_edges WHERE project = ?1",
                [project],
                |row| row.get(0),
            )
            .map_err(|error| Error::Store(format!("count persisted Delta edges: {error}")))?;
        if existing_edges != 0 {
            return Err(Error::Invalid(
                "pre-PR138 Store-CoW graph has resolved edges but no persisted raw edges to repair"
                    .into(),
            ));
        }
    }
    greppy_indexer::recover_persisted_rust_usages(&mut overlay, project, root)?;
    greppy_indexer::rebuild_visible_overlay_edges(&mut overlay, project)?;
    overlay
        .conn()
        .execute(
            "INSERT INTO main.schema_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (
                RUST_CALLER_EDGES_REPAIR_META_KEY,
                RUST_CALLER_EDGES_REPAIR_COMPLETE,
            ),
        )
        .map_err(|error| {
            Error::Store(format!("persist Rust caller-edge repair marker: {error}"))
        })?;
    Ok(true)
}

/// Repair an already-indexed single/private store on normal query open.
/// This shares the one-shot resolver marker with CoW, but never attaches or
/// scans an immutable Base. The writer lock covers re-check through commit.
pub(crate) fn ensure_persisted_single_store_repaired(
    path: &Path,
    root: &Path,
    project: &str,
) -> Result<()> {
    let observed = greppy_store::Store::open_with(path, greppy_store::OpenOptions::read_only())?;
    if !persisted_v7_delta_needs_repair(&observed, root)? {
        return Ok(());
    }
    drop(observed);
    let deadline = std::time::Instant::now() + crate::NAV_FRESHNESS_BUDGET;
    let _lock = loop {
        match greppy_freshness::try_acquire(path) {
            Ok(lock) => break lock,
            Err(greppy_freshness::LockError::Held { path: lock_path }) => {
                let observed =
                    greppy_store::Store::open_with(path, greppy_store::OpenOptions::read_only())?;
                if !persisted_v7_delta_needs_repair(&observed, root)? {
                    return Ok(());
                }
                if std::time::Instant::now() >= deadline {
                    return Err(Error::Lock(format!(
                        "timed out waiting for single-store Rust repair; lock {}",
                        lock_path.display()
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(error) => return Err(error.into()),
        }
    };
    let mut store =
        greppy_store::Store::open_with(path, greppy_store::OpenOptions::query_writer())?;
    if persisted_v7_delta_needs_repair(&store, root)? {
        if rust_repair_requires_source_refresh(&store, root, project) {
            return Ok(());
        }
        greppy_indexer::rebuild_single_store_rust_edges(&mut store, project)?;
    }
    Ok(())
}

pub(crate) fn ensure_persisted_v7_delta_repaired(
    delta_path: &Path,
    base_path: &Path,
    visibility: &VisibilityIndex,
    root: &Path,
    project: &str,
) -> Result<()> {
    let delta = greppy_store::Store::open_with(delta_path, greppy_store::OpenOptions::read_only())?;
    if !persisted_v7_delta_needs_repair(&delta, root)? {
        return Ok(());
    }
    drop(delta);
    repair_persisted_v7_delta(delta_path, base_path, visibility, root, project)?;
    let repaired =
        greppy_store::Store::open_with(delta_path, greppy_store::OpenOptions::read_only())?;
    if persisted_v7_delta_needs_repair(&repaired, root)? {
        let visible = repaired.attach_overlay(base_path, visibility)?;
        if rust_repair_requires_source_refresh(&visible, root, project) {
            return Ok(());
        }
        return Err(Error::Lock(
            "persisted Delta repair did not publish its completion marker".into(),
        ));
    }
    Ok(())
}

pub(crate) fn persisted_v7_delta_needs_repair(
    delta: &greppy_store::Store,
    root: &Path,
) -> Result<bool> {
    let marker = match delta.conn().query_row(
        "SELECT value FROM main.schema_meta WHERE key = ?1",
        [RUST_CALLER_EDGES_REPAIR_META_KEY],
        |row| row.get::<_, String>(0),
    ) {
        Ok(value) => Some(value),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(error) => {
            return Err(Error::Store(format!(
                "read Rust caller-edge repair marker: {error}"
            )))
        }
    };
    Ok(delta
        .list_private_workspace_states()?
        .into_iter()
        .any(|state| {
            let indexer_base = state
                .indexer_version
                .split_once(';')
                .map_or(state.indexer_version.as_str(), |(base, _)| base);
            paths_resolve_equal(Path::new(&state.root_path), root)
                && indexer_base == greppy_core::INDEXER_VERSION_BASE
        })
        && marker.as_deref() != Some(RUST_CALLER_EDGES_REPAIR_COMPLETE))
}

pub(crate) fn mark_rust_caller_edges_repaired(store: &greppy_store::Store) -> Result<()> {
    if store.is_overlay() && !greppy_indexer::rust_caller_edges_repaired(store)? {
        let base_marker = store.conn().query_row(
            "SELECT value FROM greppy_base.schema_meta WHERE key = ?1",
            [RUST_CALLER_EDGES_REPAIR_META_KEY],
            |row| row.get::<_, String>(0),
        );
        let base_current = match base_marker {
            Ok(value) => value == RUST_CALLER_EDGES_REPAIR_COMPLETE,
            Err(rusqlite::Error::QueryReturnedNoRows) => false,
            Err(error) => {
                return Err(Error::Store(format!(
                    "read Base Rust repair marker: {error}"
                )))
            }
        };
        if !base_current {
            // A sparse Delta rebuild does not certify an older immutable
            // Base. Leave repair pending for its one-shot visible raw pass.
            return Ok(());
        }
    }
    store
        .conn()
        .execute(
            "INSERT INTO main.schema_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (
                RUST_CALLER_EDGES_REPAIR_META_KEY,
                RUST_CALLER_EDGES_REPAIR_COMPLETE,
            ),
        )
        .map_err(|error| {
            Error::Store(format!("persist Rust caller-edge repair marker: {error}"))
        })?;
    Ok(())
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
    let persisted_binding = if binding_is_persisted {
        persisted_overlay_binding(root).ok().flatten()
    } else {
        None
    };
    let configured_mode = std::env::var(ENV_MODE).ok().or_else(|| {
        resolved_binding
            .as_ref()
            .ok()
            .and_then(|binding| binding.as_ref())
            .map(|_| MODE_OVERLAY.to_string())
            .or_else(|| persisted_binding.as_ref().map(|_| MODE_OVERLAY.to_string()))
            .or_else(|| {
                (binding_is_persisted && resolved_binding.is_err())
                    .then(|| MODE_OVERLAY.to_string())
            })
    });
    let fallback_reason = std::env::var(ENV_FALLBACK_REASON).ok();
    let base_commit = std::env::var(ENV_BASE_COMMIT).ok().or_else(|| {
        resolved_binding
            .as_ref()
            .ok()
            .and_then(|binding| binding.as_ref())
            .map(|(_, commit)| commit.clone())
            .or_else(|| {
                persisted_binding
                    .as_ref()
                    .map(|binding| binding.base_commit.clone())
            })
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

    let mut base_path = std::env::var_os(ENV_BASE_PATH)
        .map(PathBuf::from)
        .or_else(|| {
            persisted_binding
                .as_ref()
                .map(|binding| binding.base_path.clone())
        });
    let mut base_identity = None;
    let mut base_complete = base_path.as_ref().map(|path| path.is_file());
    let mut dirty_paths = None;
    let mut deleted_paths = None;
    let mut delta_identity = None;
    let mut error = resolved_binding
        .as_ref()
        .err()
        .map(|issue| issue.to_string());
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
    deadline: Option<std::time::Instant>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<BaseBuilderLease> {
    loop {
        if cancel.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::Lock(format!(
                "cancelled while waiting for immutable Base {identity_hash} publication"
            )));
        }
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            let lock_path = layout
                .builder_lock_path()
                .map_err(|error| Error::io("resolve Base builder lock", error))?;
            return Err(Error::Lock(format!(
                "deadline reached while waiting for immutable Base {identity_hash} publication; lock {}",
                lock_path.display()
            )));
        }
        if let Some(lease) = layout
            .acquire_builder(true)
            .map_err(|error| Error::io("acquire Base Store builder lease", error))?
        {
            return Ok(lease);
        }
        report_base_phase(progress_path, "waiting_for_base_builder");
        std::thread::sleep(std::time::Duration::from_millis(250));
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
        greppy_core::cache::ENV_BASE_BUILD_STAGING_LEASES,
    ];
    let restore = names
        .into_iter()
        .map(|name| (name, std::env::var_os(name)))
        .collect::<Vec<_>>();
    std::env::set_var(greppy_core::PROJECT_IDENTITY_ENV, &project);
    std::env::set_var(ENV_DISABLE_AUTO_LINKED_WORKTREE, "1");

    let structural_first_use = std::env::var_os(crate::ENV_STRUCTURAL_FIRST_USE).is_some();
    let outcome = (|| {
        // Keep an existing worktree pinned to its verified Base. Advancing the
        // primary checkout must not force every already-indexed worktree to
        // build a new repository-wide Base on its next Delta refresh.
        let existing_binding = overlay_environment_for_recovery(root)?;
        let missing_bound_graph = existing_binding
            .as_ref()
            .is_some_and(|(path, _)| !path.is_file());
        let base_commit = match existing_binding.as_ref() {
            Some((_, commit)) => commit.clone(),
            None => git_output(&primary, &["rev-parse", "HEAD"])?,
        };
        let prepared =
            match reuse_verified_base_store(&primary, &base_commit, shared_data_root, &project)? {
                Some(prepared) => Some(prepared),
                None if structural_first_use
                    && !missing_bound_graph
                    && !has_verified_previous_indexer_base(
                        &primary,
                        &base_commit,
                        shared_data_root,
                    )? =>
                {
                    None
                }
                None => {
                    // Only the first worktree for this immutable Git tree needs a
                    // clean materialization. Every later worktree opens the
                    // hash-verified published Base directly. When recovery began
                    // from a stale Delta binding, rebuild that binding's pinned
                    // commit rather than silently moving it to the primary HEAD.
                    report_base_phase(progress_path, "preparing_base_checkout");
                    let clean = TemporaryBaseWorktree::create(&primary, &base_commit)?;
                    let mut inherited_leases =
                        std::env::var_os(greppy_core::cache::ENV_BASE_BUILD_STAGING_LEASES)
                            .map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
                            .unwrap_or_default();
                    inherited_leases.push(clean.lease_root().to_path_buf());
                    let inherited_leases =
                        std::env::join_paths(inherited_leases).map_err(|error| {
                            Error::Invalid(format!(
                                "cannot pass temporary Base checkout lease to index child: {error}"
                            ))
                        })?;
                    std::env::set_var(
                        greppy_core::cache::ENV_BASE_BUILD_STAGING_LEASES,
                        inherited_leases,
                    );
                    Some(prepare_base_store_paths(
                        &primary,
                        clean.path(),
                        clean.path(),
                        &base_commit,
                        shared_data_root,
                        embedding_args,
                        progress_path,
                        None,
                        None,
                    )?)
                }
            };
        if let Some(prepared) = prepared.as_ref() {
            configure_overlay_environment(prepared, &base_commit);
            eprintln!(
                "greppy index: linked worktree uses shared Base {} at {} ({}); only the Git/dirty Delta will be indexed",
                &prepared.identity_hash[..12],
                base_commit,
                if prepared.reused { "reused" } else { "created" },
            );
        }
        Ok::<_, Error>(prepared)
    })();

    match outcome {
        Ok(Some(prepared)) => Ok(Some(AutoLinkedWorktreeOverlay {
            _prepared: prepared,
            restore,
        })),
        Ok(None) => {
            for (name, value) in restore.into_iter().rev() {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Ok(None)
        }
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
    _lease: greppy_core::cache::FileLock,
}

impl TemporaryBaseWorktree {
    fn create(primary: &Path, base_commit: &str) -> Result<Self> {
        #[cfg(debug_assertions)]
        if std::env::var_os(ENV_TEST_FORBID_TEMP_BASE_CHECKOUT).is_some() {
            return Err(Error::Invalid(
                "test forbids a second temporary Base checkout".into(),
            ));
        }
        let scratch_root = temporary_base_checkout_root()?;
        // A killed Base builder can leave its disposable checkout behind. Keep
        // the existing lease-aware reclamation after moving these directories
        // away from the persistent Base Store root.
        let _ = greppy_core::cache::reap_stale_base_build_dirs(
            &scratch_root,
            greppy_core::cache::BASE_BUILD_STAGING_TTL,
        );
        let parent = tempfile::Builder::new()
            .prefix("greppy-linked-base-checkout-")
            .tempdir_in(&scratch_root)
            .map_err(|error| {
                Error::io(
                    format!(
                        "create clean Base checkout under scratch directory {}",
                        scratch_root.display()
                    ),
                    error,
                )
            })?;
        let lease = greppy_core::cache::create_base_build_staging_lease(parent.path())
            .map_err(|error| Error::io("lease clean Base checkout", error))?;
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
            _lease: lease,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn lease_root(&self) -> &Path {
        self._parent.path()
    }
}

fn temporary_base_checkout_root() -> Result<PathBuf> {
    // Honor TMPDIR consistently on every platform. Rust's Windows
    // `temp_dir()` follows GetTempPath and would otherwise ignore an explicit
    // scratch directory supplied by the caller.
    let root = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    if !root.is_absolute() {
        return Err(Error::Invalid(format!(
            "temporary Base checkout directory must be absolute: {}",
            root.display()
        )));
    }
    let metadata = std::fs::metadata(&root).map_err(|error| {
        Error::io(
            format!(
                "inspect temporary Base checkout directory {}; set TMPDIR to an existing writable scratch directory",
                root.display()
            ),
            error,
        )
    })?;
    if !metadata.is_dir() {
        return Err(Error::Invalid(format!(
            "temporary Base checkout directory is not a directory: {}; set TMPDIR to an existing writable scratch directory",
            root.display()
        )));
    }
    Ok(root)
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

/// Return whether a verified v6 Base is available for the current immutable
/// inputs. Structural first use may skip a cold Base build, but it must still
/// migrate an existing v6 artifact before publishing a v7 Delta: PR131's Rust
/// extraction and resolution changes are not safe to hide behind a freshness
/// proof over the old graph.
fn has_verified_previous_indexer_base(
    repo_root: &Path,
    base_commit: &str,
    shared_data_root: &Path,
) -> Result<bool> {
    let current_identity = base_identity_parts(repo_root, base_commit)?;
    if current_identity.indexer_version != "greppy-indexer-v7" {
        return Ok(false);
    }
    return has_verified_previous_indexer_base_for_identity(shared_data_root, &current_identity);
}

fn has_verified_previous_indexer_base_for_identity(
    shared_data_root: &Path,
    current_identity: &BaseStoreIdentity,
) -> Result<bool> {
    if current_identity.indexer_version != "greppy-indexer-v7" {
        return Ok(false);
    }
    let mut previous_identity = current_identity.clone();
    previous_identity.indexer_version = "greppy-indexer-v6".into();
    let layout = BaseStoreLayout::new(shared_data_root, &previous_identity)
        .map_err(|error| Error::io("construct previous Base Store layout", error))?;
    Ok(layout
        .read_verified_manifest()
        .is_ok_and(|manifest| manifest.identity == previous_identity))
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
    let paths = compatible_worktree_paths(
        || {
            let mut command = Command::new("git");
            command
                .arg("-C")
                .arg(root)
                .args(["worktree", "list", "--porcelain", "-z"]);
            let output = command
                .output()
                .map_err(|error| Error::io("list linked Git worktrees", error))?;
            if output.status.success() {
                Ok(output.stdout)
            } else {
                Err(Error::Invalid(
                    String::from_utf8_lossy(&output.stderr).trim().to_string(),
                ))
            }
        },
        || canonical_repository_common_dir(root),
    )?;
    for candidate in paths {
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

fn compatible_worktree_paths(
    list_nul: impl FnOnce() -> Result<Vec<u8>>,
    common_dir: impl FnOnce() -> Result<PathBuf>,
) -> Result<Vec<PathBuf>> {
    match list_nul() {
        Ok(output) => parse_nul_worktree_paths(&output),
        Err(nul_error) => {
            let common_dir = common_dir().map_err(|common_error| {
                Error::Invalid(format!(
                    "cannot list linked Git worktrees with NUL porcelain output: {nul_error}; cannot resolve the common Git directory: {common_error}"
                ))
            })?;
            if common_dir.file_name().and_then(|name| name.to_str()) != Some(".git") {
                return Err(Error::Invalid(format!(
                    "cannot list linked Git worktrees with NUL porcelain output: {nul_error}; common Git directory {} does not identify a primary checkout",
                    common_dir.display()
                )));
            }
            let primary = common_dir.parent().ok_or_else(|| {
                Error::Invalid(format!(
                    "common Git directory {} has no parent checkout",
                    common_dir.display()
                ))
            })?;
            Ok(vec![primary.to_path_buf()])
        }
    }
}

fn parse_nul_worktree_paths(output: &[u8]) -> Result<Vec<PathBuf>> {
    output
        .split(|byte| *byte == 0)
        .filter_map(|field| field.strip_prefix(b"worktree "))
        .map(|path| {
            std::str::from_utf8(path)
                .map(PathBuf::from)
                .map_err(|_| Error::Invalid("Git worktree path is not valid UTF-8".into()))
        })
        .collect()
}

pub(crate) fn prepare_base_store(
    workspace: &greppy_agent::workspace::AgentWorkspace,
    shared_data_root: &Path,
    embedding_args: crate::EmbeddingCliArgs<'_>,
    deadline: Option<std::time::Instant>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PreparedBase> {
    prepare_base_store_paths(
        workspace.repo_root(),
        workspace.repository_path(),
        workspace.worktree_path(),
        workspace.base_commit(),
        shared_data_root,
        embedding_args,
        None,
        deadline,
        cancel,
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
    deadline: Option<std::time::Instant>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<PreparedBase> {
    let structural_first_use = std::env::var_os(crate::ENV_STRUCTURAL_FIRST_USE).is_some();
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

    // Poll rather than blocking in flock so progress remains observable. A
    // matching live builder owns publication; wait for its OS lock to release,
    // then validate and reuse its completed Base below. If it dies or fails,
    // the same lock release elects this caller as the replacement builder.
    let builder_lease =
        acquire_base_builder(&layout, &identity_hash, progress_path, deadline, cancel)?;
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
    // Builds that died (ENOSPC, OOM, SIGKILL) leave their staging directory
    // and temporary checkout behind; 44 of them held 23 GB on one machine.
    let _ = greppy_core::cache::reap_stale_base_build_dirs(
        shared_data_root,
        greppy_core::cache::BASE_BUILD_STAGING_TTL,
    );
    let staging = tempfile::Builder::new()
        .prefix("greppy-base-build-")
        .tempdir_in(shared_data_root)
        .map_err(|error| Error::io("create Base build staging directory", error))?;
    let _staging_lease = greppy_core::cache::create_base_build_staging_lease(staging.path())
        .map_err(|error| Error::io("lease Base build staging directory", error))?;
    let mut lease_roots = vec![std::fs::canonicalize(staging.path())
        .map_err(|error| Error::io("resolve Base staging lease", error))?];
    // Git may return the canonical /private/... spelling while the configured
    // shared root uses /tmp/... (or another directory alias). Compare the same
    // namespace so the child's checkout lease is not silently omitted.
    let canonical_shared = std::fs::canonicalize(shared_data_root)
        .map_err(|error| Error::io("resolve shared Base staging root", error))?;
    let canonical_worktree = std::fs::canonicalize(worktree_path)
        .map_err(|error| Error::io("resolve Base staging worktree", error))?;
    if let Some(checkout) = canonical_worktree.ancestors().find(|ancestor| {
        ancestor.parent() == Some(canonical_shared.as_path())
            && ancestor
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("greppy-linked-base-checkout-"))
    }) {
        lease_roots.push(
            std::fs::canonicalize(checkout)
                .map_err(|error| Error::io("resolve Base checkout lease", error))?,
        );
    }
    if let Some(inherited) = std::env::var_os(greppy_core::cache::ENV_BASE_BUILD_STAGING_LEASES) {
        lease_roots.extend(std::env::split_paths(&inherited));
    }
    lease_roots.sort();
    lease_roots.dedup();
    let child_leases = std::env::join_paths(&lease_roots)
        .map_err(|error| Error::Invalid(format!("cannot pass Base staging leases: {error}")))?;
    let staging_data = staging.path().join("data");
    std::fs::create_dir_all(&staging_data)
        .map_err(|error| Error::io("create Base build data directory", error))?;
    let staged_graph = staging_data
        .join("workspaces")
        .join(format!("v{}", greppy_core::cache::STORE_FORMAT_VERSION))
        .join(greppy_core::workspace_hash(worktree_path))
        .join("graph.db");
    let seeded_summary_cache =
        seed_previous_indexer_base(shared_data_root, &identity, worktree_path, &staged_graph)?;
    let defer_base_embeddings = structural_first_use;
    if seeded_summary_cache.is_some() {
        report_base_phase(progress_path, "migrating_base_graph");
    }
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
    if defer_base_embeddings {
        command.env(crate::ENV_STRUCTURAL_FIRST_USE, "1");
    } else {
        command.env_remove(crate::ENV_STRUCTURAL_FIRST_USE);
    }
    command
        .current_dir(worktree_path)
        .env("GREPPY_STORE_DIR", &staging_data)
        .env(
            greppy_core::cache::ENV_BASE_BUILD_STAGING_LEASES,
            child_leases,
        )
        // The staging store isolates the graph, not the inference artifacts:
        // without this the child opened an empty content cache under the
        // staging root, re-embedded every span of the repository for each
        // linked worktree (41k spans, ~35 min on this Mac for a tree whose
        // primary checkout was already fully embedded) and copied the model
        // into the staging directory as well.
        .env(
            greppy_core::cache::ENV_SHARED_INFERENCE_ROOT,
            greppy_core::cache::shared_inference_root(),
        )
        // A normal Base build completes every candidate before publication;
        // never let the ordinary foreground-index lazy threshold hand it to a
        // background process outside the publication lease. Structural Base
        // recovery and migration inherit GREPPY_STRUCTURAL_FIRST_USE even
        // without a reusable seed; they record an exact deferred receipt below
        // for later semantic completion instead of loading inference models.
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
        .env(crate::ENV_BASE_BUILD_OWNER_STDIN, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::null());
    if let Some(path) = progress_path {
        command.env(crate::ENV_DELEGATED_BACKGROUND_JOB, path);
    }
    crate::begin_delegated_base_owner();
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            crate::clear_delegated_base_owner();
            return Err(Error::io("start immutable Base index build", error));
        }
    };
    // Child::wait closes a still-attached stdin. Take the pipe and retain its
    // writer explicitly so EOF means that this owner died, not that it waited.
    let owner_writer = match child.stdin.take() {
        Some(owner) => owner,
        None => {
            crate::clear_delegated_base_owner();
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::Invalid(
                "immutable Base index build has no owner pipe".into(),
            ));
        }
    };
    crate::register_delegated_base_owner(owner_writer);
    let status = child.wait();
    crate::clear_delegated_base_owner();
    let status = status.map_err(|error| Error::io("wait for immutable Base index build", error))?;
    if !status.success() {
        return Err(Error::Invalid(format!(
            "immutable Base index build exited {status}"
        )));
    }
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
    if defer_base_embeddings {
        mark_base_embeddings_deferred(
            &staged_graph,
            &greppy_core::project_identity(worktree_path),
            &identity,
        )?;
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
    let staged_summary_cache = match seeded_summary_cache {
        Some(path) => path,
        None => {
            build_base_summary_cache(&staged_graph, &identity.summary_model_and_prompt_version)?
        }
    };
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

fn base_embedding_deferred_key(project: &str) -> String {
    format!("{BASE_EMBEDDING_DEFERRED_META_PREFIX}{project}")
}

fn mark_base_embeddings_deferred(
    graph_path: &Path,
    project: &str,
    identity: &BaseStoreIdentity,
) -> Result<()> {
    let store =
        greppy_store::Store::open_with(graph_path, greppy_store::OpenOptions::query_writer())?;
    let generation = store
        .list_workspace_states()?
        .into_iter()
        .map(|state| state.graph_generation)
        .max()
        .ok_or_else(|| Error::Invalid("Base build has no workspace generation".into()))?;
    store
        .conn()
        .execute(
            "INSERT INTO schema_meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            rusqlite::params![
                base_embedding_deferred_key(project),
                format!("{generation}|{}", identity.embedding_model)
            ],
        )
        .map_err(|error| Error::Store(format!("record deferred Base embeddings: {error}")))?;
    store
        .conn()
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .map_err(|error| Error::Store(format!("checkpoint deferred Base receipt: {error}")))?;
    Ok(())
}

/// Seed a v7 Base build from the verified v6 artifact with the same immutable
/// inputs. The old published Base remains untouched; the ordinary index
/// command opens this private copy and performs the scoped raw-edge migration.
/// Since byte-identical nodes and vectors survive that migration, the child
/// does not need to re-run model inference.
fn seed_previous_indexer_base(
    shared_data_root: &Path,
    current_identity: &BaseStoreIdentity,
    worktree_path: &Path,
    staged_graph: &Path,
) -> Result<Option<PathBuf>> {
    if current_identity.indexer_version != "greppy-indexer-v7" {
        return Ok(None);
    }
    let mut previous_identity = current_identity.clone();
    previous_identity.indexer_version = "greppy-indexer-v6".into();
    let previous_layout = BaseStoreLayout::new(shared_data_root, &previous_identity)
        .map_err(|error| Error::io("construct previous Base Store layout", error))?;
    let Ok(previous_manifest) = previous_layout.read_verified_manifest() else {
        return Ok(None);
    };
    if previous_manifest.identity != previous_identity {
        return Ok(None);
    }
    let parent = staged_graph
        .parent()
        .ok_or_else(|| Error::Invalid("staged Base graph has no parent directory".into()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| Error::io("create migrated Base graph directory", error))?;
    let mut previous_graph = std::fs::File::open(&previous_layout.graph)
        .map_err(|error| Error::io("open previous Base graph for migration", error))?;
    let mut migrated_graph = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(staged_graph)
        .map_err(|error| Error::io("create migrated Base graph", error))?;
    std::io::copy(&mut previous_graph, &mut migrated_graph)
        .map_err(|error| Error::io("copy previous Base graph for migration", error))?;
    drop(migrated_graph);
    // The indexer keys workspace compatibility by its canonical repository
    // root. Temporary checkouts can have a different lexical spelling on
    // macOS (`/var/...` versus `/private/var/...`); persisting the lexical
    // path makes the v6 workspace lookup miss, forcing a full rebuild whose
    // node cascade discards otherwise reusable vectors.
    let canonical_worktree = std::fs::canonicalize(worktree_path)
        .map_err(|error| Error::io("resolve migrated Base worktree", error))?;
    let root = canonical_worktree.to_string_lossy();
    let store =
        greppy_store::Store::open_with(staged_graph, greppy_store::OpenOptions::query_writer())?;
    store
        .conn()
        .execute(
            "UPDATE main.projects SET root_path = ?1",
            rusqlite::params![root.as_ref()],
        )
        .map_err(|error| Error::Store(format!("retarget migrated Base project: {error}")))?;
    store
        .conn()
        .execute(
            "UPDATE main.workspace_state SET root_path = ?1",
            rusqlite::params![root.as_ref()],
        )
        .map_err(|error| Error::Store(format!("retarget migrated Base workspace: {error}")))?;
    let staged_summary_cache = parent
        .join("base-summary-cache")
        .join(greppy_store::SUMMARY_CACHE_DB_FILE);
    let summary_parent = staged_summary_cache.parent().ok_or_else(|| {
        Error::Invalid("staged Base summary cache has no parent directory".into())
    })?;
    std::fs::create_dir_all(summary_parent)
        .map_err(|error| Error::io("create migrated Base summary directory", error))?;
    let mut previous_summary = std::fs::File::open(&previous_layout.summary_cache)
        .map_err(|error| Error::io("open previous Base summary cache", error))?;
    let mut migrated_summary = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged_summary_cache)
        .map_err(|error| Error::io("create migrated Base summary cache", error))?;
    std::io::copy(&mut previous_summary, &mut migrated_summary)
        .map_err(|error| Error::io("copy previous Base summary cache", error))?;
    drop(migrated_summary);
    Ok(Some(staged_summary_cache))
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
    // A structurally migrated Base is immutable and safe for graph queries
    // before semantic completion. Accept only the receipt written after that
    // controlled migration, bound to the same generation and model identity;
    // an arbitrary missing or stale completion marker still fails closed.
    let deferred: Option<String> = store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [base_embedding_deferred_key(project)],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| Error::Store(format!("read deferred Base embedding receipt: {error}")))?;
    #[cfg(debug_assertions)]
    let injected_summary_failure = std::env::var_os(ENV_TEST_BASE_SUMMARY_FAIL).is_some();
    #[cfg(not(debug_assertions))]
    let injected_summary_failure = false;
    if !injected_summary_failure
        && !base_embedding_receipt_valid(
            completion.as_deref(),
            deferred.as_deref(),
            &expected_completion,
        )
    {
        return Err(Error::Invalid(format!(
            "Base embedding generation is incomplete: expected completion or deferred receipt `{expected_completion}`, got completion={} deferred={}",
            completion.as_deref().unwrap_or("missing"),
            deferred.as_deref().unwrap_or("missing")
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

fn base_embedding_receipt_valid(
    completion: Option<&str>,
    deferred: Option<&str>,
    expected: &str,
) -> bool {
    completion == Some(expected) || deferred == Some(expected)
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
    let common_path = canonical_repository_common_dir(repo)?;
    Ok(format!("git-common-dir:{}", common_path.display()))
}

fn canonical_repository_common_dir(repo: &Path) -> Result<PathBuf> {
    let path = git_path_output(
        repo,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )
    .or_else(|_| git_path_output(repo, &["rev-parse", "--git-common-dir"]))?;
    let absolute = if path.is_absolute() {
        path
    } else {
        repo.join(path)
    };
    Ok(absolute.canonicalize().unwrap_or(absolute))
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

fn git_path_output(root: &Path, args: &[&str]) -> Result<PathBuf> {
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
    let mut value = output.stdout;
    if value.last() == Some(&b'\n') {
        value.pop();
        if value.last() == Some(&b'\r') {
            value.pop();
        }
    }
    if value.is_empty() {
        return Err(Error::Invalid(format!(
            "git {} returned empty output",
            args.join(" ")
        )));
    }
    let value = String::from_utf8(value)
        .map_err(|_| Error::Invalid(format!("git {} returned non-UTF-8", args.join(" "))))?;
    Ok(PathBuf::from(value))
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
    use clap::Parser;

    struct TmpdirRestore(Option<std::ffi::OsString>);

    impl TmpdirRestore {
        fn set(path: &Path) -> Self {
            let previous = std::env::var_os("TMPDIR");
            std::env::set_var("TMPDIR", path);
            Self(previous)
        }
    }

    impl Drop for TmpdirRestore {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var("TMPDIR", value),
                None => std::env::remove_var("TMPDIR"),
            }
        }
    }

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

    struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);

    impl EnvRestore {
        fn capture(names: &[&'static str]) -> Self {
            Self(
                names
                    .iter()
                    .map(|name| (*name, std::env::var_os(name)))
                    .collect(),
            )
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (name, value) in &self.0 {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    fn assert_persisted_v7_delta_query_is_correct(root: &str) -> std::result::Result<(), String> {
        let cli = crate::Cli::try_parse_from(["greppy", "--root", root, "who-calls", "target"])
            .map_err(|error| error.to_string())?;
        let exit = crate::dispatch(cli).map_err(|error| error.to_string())?;
        if exit != 0 {
            return Err(format!("who-calls CLI returned exit code {exit}"));
        }
        let effective_root = Path::new(root);
        let overlay = overlay_spec(effective_root)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "overlay configuration disappeared".to_string())?;
        let delta_path = crate::workspace_locator::store_path(effective_root);
        let store =
            greppy_store::Store::open_overlay(&overlay.base_path, &delta_path, &overlay.visibility)
                .map_err(|error| error.to_string())?;
        let target = store
            .get_node_by_qname("p", "src/alias_chain/sub.rs::Function::target")
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "repaired target node is missing".to_string())?;
        let caller = store
            .get_node_by_qname("p", "src/caller.rs::Function::caller")
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "repaired caller node is missing".to_string())?;
        if !store
            .incoming_edges(target.id, Some("USAGE"), 10)
            .map_err(|error| error.to_string())?
            .iter()
            .any(|edge| edge.source_id == caller.id)
        {
            return Err("repaired caller edge is missing".into());
        }
        Ok(())
    }

    #[test]
    fn sparse_cow_marker_does_not_certify_an_unrepaired_base() {
        let scratch = tempfile::tempdir().unwrap();
        let repo = fixture();
        let base_path = scratch.path().join("base.db");
        let delta_path = scratch.path().join("delta.db");
        {
            let mut base = greppy_store::Store::open(&base_path).unwrap();
            greppy_indexer::index(&mut base, repo.path(), "p").unwrap();
            base.conn()
                .execute(
                    "DELETE FROM schema_meta WHERE key=?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                )
                .unwrap();
        }
        let visibility = VisibilityIndex::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();
        let store =
            greppy_store::Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        mark_rust_caller_edges_repaired(&store).unwrap();
        assert!(
            !greppy_indexer::rust_caller_edges_repaired(&store).unwrap(),
            "Delta-only work cannot mark a legacy Base complete"
        );
        drop(store);
        let base =
            greppy_store::Store::open_with(&base_path, greppy_store::OpenOptions::query_writer())
                .unwrap();
        greppy_indexer::mark_rust_caller_edges_repaired(&base).unwrap();
        drop(base);
        let store =
            greppy_store::Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        mark_rust_caller_edges_repaired(&store).unwrap();
        assert!(
            greppy_indexer::rust_caller_edges_repaired(&store).unwrap(),
            "a current Base makes sparse publication current without a Base pass"
        );
    }

    #[test]
    fn persisted_rust_repair_defers_to_refresh_after_source_edits() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let _env = EnvRestore::capture(&["GREPPY_STORE_DIR", "GREPPY_PROJECT_IDENTITY", "GREPPY_AUTO_REINDEX", "GREPPY_TEST_SKIP_INFERENCE", ENV_MODE, ENV_BASE_PATH, ENV_BASE_COMMIT]);
                for name in [ENV_MODE, ENV_BASE_PATH, ENV_BASE_COMMIT] { std::env::remove_var(name); }
                let scratch = tempfile::tempdir().unwrap();
                let repo = fixture();
                let root = crate::resolving::resolve_root(Some(&repo.path().to_string_lossy())).unwrap();
                let root_string = root.to_string_lossy().into_owned();
                std::env::set_var("GREPPY_STORE_DIR", scratch.path().join("store"));
                std::env::set_var("GREPPY_PROJECT_IDENTITY", "p");
                std::env::set_var("GREPPY_AUTO_REINDEX", "0");
                std::env::set_var("GREPPY_TEST_SKIP_INFERENCE", "1");
                let path = crate::workspace_locator::store_path(&root);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                let mut store = greppy_store::Store::open(&path).unwrap();
                greppy_indexer::index(&mut store, &root, "p").unwrap();
                store.conn().execute("DELETE FROM schema_meta WHERE key=?1", [RUST_CALLER_EDGES_REPAIR_META_KEY]).unwrap();
                let original_states = format!("{:?}", store.list_file_states("p").unwrap());
                drop(store);
                std::fs::write(root.join("src/a.rs"), "pub fn refreshed_target() {}\npub fn refreshed_caller() { refreshed_target(); }\n").unwrap();
                for store in [crate::freshness::open_default_store(Some(&root_string)).unwrap(), crate::freshness::open_default_store_query_writer(Some(&root_string)).unwrap()] {
                    assert!(persisted_v7_delta_needs_repair(&store, &root).unwrap());
                    assert_eq!(format!("{:?}", store.list_file_states("p").unwrap()), original_states);
                    let proof = crate::nav_freshness_json_uncached(&store, Some(&root_string), "p");
                    assert!(crate::freshness::freshness_is_reindexable_stale(&proof), "{proof:?}");
                    assert!(matches!(crate::freshness::freshness_serve_decision(&store, Some(&root_string), "p"), crate::FreshnessServe::Refuse(_)), "deferred migration must not authorize a stale graph");
                }
                // Exercise the same atomic publication used by the automatic
                // query refresh, without spawning the unit-test executable.
                crate::indexing::index_atomic_snapshot(&path, &root, "p", None, &greppy_indexer::IndexOptions::default(), false, None).unwrap();
                let store = crate::freshness::open_default_store(Some(&root_string)).unwrap();
                assert!(greppy_indexer::rust_caller_edges_repaired(&store).unwrap());
                let target = store.get_node_by_qname("p", "src/a.rs::Function::refreshed_target").unwrap().unwrap();
                let caller = store.get_node_by_qname("p", "src/a.rs::Function::refreshed_caller").unwrap().unwrap();
                assert!(store.incoming_edges(target.id, Some("CALLS"), 20).unwrap().iter().any(|edge| edge.source_id == caller.id));
                assert!(store.get_node_by_qname("p", "src/a.rs::Function::a").unwrap().is_none());
            }).unwrap().join().unwrap();
    }

    #[test]
    fn persisted_single_store_rust_repair_preserves_cache_and_is_one_shot() {
        let test = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(persisted_single_store_rust_repair_preserves_cache_and_is_one_shot_body)
            .unwrap();
        if let Err(panic) = test.join() {
            std::panic::resume_unwind(panic);
        }
    }

    fn persisted_single_store_rust_repair_preserves_cache_and_is_one_shot_body() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _env = EnvRestore::capture(&[
            "GREPPY_STORE_DIR",
            "GREPPY_PROJECT_IDENTITY",
            "GREPPY_AUTO_REINDEX",
            "GREPPY_TEST_SKIP_INFERENCE",
            ENV_MODE,
            ENV_BASE_PATH,
            ENV_BASE_COMMIT,
        ]);
        for name in [ENV_MODE, ENV_BASE_PATH, ENV_BASE_COMMIT] {
            std::env::remove_var(name);
        }
        let scratch = tempfile::tempdir().unwrap();
        let repo = fixture();
        let root = crate::resolving::resolve_root(Some(&repo.path().to_string_lossy())).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"vcop2-tools\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub mod m68000_aot;\npub enum Instruction { AddImmediateByte { amount: u8 } }\npub fn decode() -> Instruction { Instruction::AddImmediateByte { amount: 1 } }\npub fn amount() {}\npub fn valid() { let _ = amount; }\n").unwrap();
        std::fs::write(root.join("src/m68000_aot.rs"), "pub fn compile() {}\n").unwrap();
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(
            root.join("tests/callers.rs"),
            "use vcop2_tools::m68000_aot::{compile};\nfn caller() { compile(); }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/other.py"),
            "def py_target():\n    pass\ndef py_caller():\n    py_target()\n",
        )
        .unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "single-store fixture"]);
        std::env::set_var("GREPPY_STORE_DIR", scratch.path().join("store"));
        std::env::set_var("GREPPY_PROJECT_IDENTITY", "p");
        std::env::set_var("GREPPY_AUTO_REINDEX", "0");
        std::env::set_var("GREPPY_TEST_SKIP_INFERENCE", "1");
        let path = crate::workspace_locator::store_path(&root);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut store = greppy_store::Store::open(&path).unwrap();
        let indexed = greppy_indexer::index(&mut store, &root, "p").unwrap();
        assert!(greppy_indexer::rust_caller_edges_repaired(&store).unwrap());
        let target = store
            .get_node_by_qname("p", "src/m68000_aot.rs::Function::compile")
            .unwrap()
            .unwrap();
        let caller = store
            .get_node_by_qname("p", "tests/callers.rs::Function::caller")
            .unwrap()
            .unwrap();
        store
            .upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                project: "p".into(),
                model_id: "fixture".into(),
                prompt_version: "fixture".into(),
                task: "code".into(),
                node_id: Some(caller.id),
                chunk_idx: 0,
                qualified_name: caller.qualified_name.clone(),
                file_path: caller.file_path.clone(),
                start_line: 2,
                end_line: 2,
                content_sha256: "a".repeat(64),
                graph_generation: indexed.graph_generation,
                vector: vec![1.0, 0.0],
            })
            .unwrap();
        let nodes_before = format!(
            "{:?}",
            store.list_nodes_by_label("p", "Function", 100).unwrap()
        );
        let states_before = format!("{:?}", store.list_file_states("p").unwrap());
        let workspace_before = format!("{:?}", store.list_workspace_states().unwrap());
        let vector_before: Vec<u8> = store
            .conn()
            .query_row(
                "SELECT vector FROM vector_embeddings WHERE project = 'p'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let py_before: i64 = store.conn().query_row("SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id=e.source_id WHERE n.file_path='src/other.py'", [], |row| row.get(0)).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE target_id=?1 AND edge_type IN ('CALLS','USAGE','IMPORTS')",
                [target.id],
            )
            .unwrap();
        store
            .insert_edge(&greppy_store::NewEdge {
                project: "p".into(),
                source_id: caller.id,
                target_id: caller.id,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({}),
            })
            .unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM schema_meta WHERE key=?1",
                [RUST_CALLER_EDGES_REPAIR_META_KEY],
            )
            .unwrap();
        let constructor = store
            .get_node_by_qname("p", "src/lib.rs::Instruction::AddImmediateByte")
            .unwrap()
            .unwrap();
        let decode = store
            .get_node_by_qname("p", "src/lib.rs::Function::decode")
            .unwrap()
            .unwrap();
        let amount = store
            .get_node_by_qname("p", "src/lib.rs::Function::amount")
            .unwrap()
            .unwrap();
        let valid = store
            .get_node_by_qname("p", "src/lib.rs::Function::valid")
            .unwrap()
            .unwrap();
        store
            .insert_raw_edges(&[greppy_store::NewRawEdge {
                project: "p".into(),
                file_path: "src/lib.rs".into(),
                source_qname: decode.qualified_name.clone(),
                target_qname: amount.qualified_name.clone(),
                edge_type: "USAGE".into(),
                properties: serde_json::json!({"ref_name": "amount", "line": 3}),
            }])
            .unwrap();
        store
            .insert_edge(&greppy_store::NewEdge {
                project: "p".into(),
                source_id: decode.id,
                target_id: amount.id,
                edge_type: "USAGE".into(),
                properties: serde_json::json!({"ref_name": "amount"}),
            })
            .unwrap();
        store.conn().execute("DELETE FROM raw_edges WHERE target_qname LIKE '%AddImmediateByte%' AND edge_type='USAGE'", []).unwrap();
        store
            .conn()
            .execute(
                "DELETE FROM edges WHERE target_id=?1 AND edge_type='USAGE'",
                [constructor.id],
            )
            .unwrap();
        store.conn().execute("INSERT OR REPLACE INTO schema_meta(key,value) VALUES('greppy.rust_caller_edges_repair.v2','complete')", []).unwrap();
        store.conn().execute_batch("CREATE TRIGGER reject_rust_repair BEFORE INSERT ON edges WHEN NEW.edge_type='CALLS' BEGIN SELECT RAISE(ABORT,'fixture repair failure'); END;").unwrap();
        drop(store);
        let root_string = root.to_string_lossy().into_owned();
        assert!(crate::freshness::open_default_store(Some(&root_string)).is_err());
        let store =
            greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())
                .unwrap();
        assert!(
            !greppy_indexer::rust_caller_edges_repaired(&store).unwrap(),
            "failed replacement cannot publish completeness"
        );
        assert_eq!(
            store
                .outgoing_edges(caller.id, Some("CALLS"), 20)
                .unwrap()
                .len(),
            1,
            "failed repair rolls back old edge deletion"
        );
        store
            .conn()
            .execute_batch("DROP TRIGGER reject_rust_repair")
            .unwrap();
        drop(store);
        let store = crate::freshness::open_default_store(Some(&root_string)).unwrap();
        assert!(store
            .incoming_edges(target.id, Some("CALLS"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        assert!(store
            .outgoing_edges(caller.id, Some("CALLS"), 20)
            .unwrap()
            .iter()
            .all(|edge| edge.target_id != caller.id));
        assert!(store
            .incoming_edges(constructor.id, Some("USAGE"), 20)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == decode.id));
        let amount_usages = store.incoming_edges(amount.id, Some("USAGE"), 20).unwrap();
        assert!(amount_usages.iter().all(|edge| edge.source_id != decode.id));
        assert!(amount_usages.iter().any(|edge| edge.source_id == valid.id));
        assert!(greppy_indexer::rust_caller_edges_repaired(&store).unwrap());
        assert_eq!(
            format!(
                "{:?}",
                store.list_nodes_by_label("p", "Function", 100).unwrap()
            ),
            nodes_before
        );
        assert_eq!(
            format!("{:?}", store.list_file_states("p").unwrap()),
            states_before
        );
        assert_eq!(
            format!("{:?}", store.list_workspace_states().unwrap()),
            workspace_before
        );
        assert_eq!(
            store
                .conn()
                .query_row(
                    "SELECT vector FROM vector_embeddings WHERE project='p'",
                    [],
                    |row| row.get::<_, Vec<u8>>(0)
                )
                .unwrap(),
            vector_before
        );
        assert_eq!(store.conn().query_row("SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id=e.source_id WHERE n.file_path='src/other.py'", [], |row| row.get::<_, i64>(0)).unwrap(), py_before);
        drop(store);
        // A trigger makes any repeated relation resolution fail, so successful
        // ordinary read and writer opens prove the marker skips that work.
        let store =
            greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())
                .unwrap();
        store.conn().execute_batch("CREATE TRIGGER reject_repeat_repair BEFORE INSERT ON edges BEGIN SELECT RAISE(ABORT,'repair repeated'); END;").unwrap();
        drop(store);
        drop(crate::freshness::open_default_store(Some(&root_string)).unwrap());
        drop(crate::freshness::open_default_store_query_writer(Some(&root_string)).unwrap());
    }

    #[test]
    fn persisted_v7_delta_repair_is_one_shot_and_preserves_vectors() {
        let test = std::thread::Builder::new()
            .name("persisted-v7-delta-repair".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(persisted_v7_delta_repair_is_one_shot_and_preserves_vectors_body)
            .expect("spawn persisted repair test on CLI-sized stack");
        if let Err(panic) = test.join() {
            std::panic::resume_unwind(panic);
        }
    }

    fn persisted_v7_delta_repair_is_one_shot_and_preserves_vectors_body() {
        let timing_start = std::time::Instant::now();
        let timing = |stage: &str| {
            eprintln!(
                "cow_repair_stage={stage} elapsed_ms={}",
                timing_start.elapsed().as_millis()
            );
        };
        timing("start");
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _env = EnvRestore::capture(&[
            "GREPPY_STORE_DIR",
            "GREPPY_PROJECT_IDENTITY",
            ENV_MODE,
            ENV_BASE_PATH,
            ENV_BASE_COMMIT,
            crate::ENV_STRUCTURAL_FIRST_USE,
        ]);
        timing("environment-lock-acquired");
        let scratch = tempfile::tempdir().unwrap();
        // The freshness proof compares the live workspace with the exact
        // pinned Git tree, so this regression must use a real repository.
        timing("fixture-start");
        let repo = fixture();
        timing("fixture-created");
        // Ordinary navigation resolves an explicit root before locating both
        // workspace state and its store. Persist the synthetic fixture under
        // that same spelling: macOS aliases /var to /private/var, and Windows
        // can similarly normalize an extended path.
        let raw_root = repo.path().to_string_lossy().into_owned();
        let root = crate::resolving::resolve_root(Some(&raw_root)).unwrap();
        std::fs::create_dir_all(root.join("src/alias_chain")).unwrap();
        std::fs::write(
            root.join("src/alias_chain/mod.rs"),
            "pub mod sub;\npub use sub::target as outer;\n",
        )
        .unwrap();
        std::fs::write(root.join("src/alias_chain/sub.rs"), "pub fn target() {}\n").unwrap();
        std::fs::write(
            root.join("src/caller.rs"),
            "use crate::alias_chain::outer;\npub fn caller() { let _ = outer; }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/base.rs"),
            "pub fn base_caller() { crate::alias_chain::sub::target(); }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/stable.rs"),
            "pub fn stable_caller() { crate::alias_chain::sub::target(); }\n",
        )
        .unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "alias base"]);
        let base_commit = git(&root, &["rev-parse", "HEAD"]);
        std::fs::write(
            root.join("src/caller.rs"),
            "use crate::alias_chain::outer;\npub fn caller() { let _ = outer; }\n// dirty Delta\n",
        )
        .unwrap();
        std::env::set_var("GREPPY_STORE_DIR", scratch.path().join("store"));
        std::env::set_var("GREPPY_PROJECT_IDENTITY", "p");
        let staged_base_path = scratch.path().join("base.db");
        let delta_path = crate::workspace_locator::store_path(&root);
        std::fs::create_dir_all(delta_path.parent().unwrap()).unwrap();

        {
            let mut base = greppy_store::Store::open(&staged_base_path).unwrap();
            base.upsert_project(&greppy_store::Project {
                name: "p".into(),
                indexed_at: "2026-09-27T00:00:00Z".into(),
                root_path: root.to_string_lossy().into_owned(),
            })
            .unwrap();
            base.insert_node(&greppy_store::NewNode {
                project: "p".into(),
                label: "Module".into(),
                name: "mod".into(),
                qualified_name: "src/alias_chain/mod.rs::__file__".into(),
                file_path: "src/alias_chain/mod.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
            let target_id = base
                .insert_node(&greppy_store::NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: "target".into(),
                    qualified_name: "src/alias_chain/sub.rs::Function::target".into(),
                    file_path: "src/alias_chain/sub.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            base.insert_node(&greppy_store::NewNode {
                project: "p".into(),
                label: "Function".into(),
                name: "base_caller".into(),
                qualified_name: "src/base.rs::Function::base_caller".into(),
                file_path: "src/base.rs".into(),
                start_line: 1,
                end_line: 1,
                properties: serde_json::json!({}),
            })
            .unwrap();
            let stable_caller_id = base
                .insert_node(&greppy_store::NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: "stable_caller".into(),
                    qualified_name: "src/stable.rs::Function::stable_caller".into(),
                    file_path: "src/stable.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            base.insert_edge(&greppy_store::NewEdge {
                project: "p".into(),
                source_id: stable_caller_id,
                target_id,
                edge_type: "CALLS".into(),
                properties: serde_json::json!({"ref_name": "target"}),
            })
            .unwrap();
            // Persist real source fingerprints for this pre-fix Base fixture.
            for rel in [
                "src/alias_chain/mod.rs",
                "src/alias_chain/sub.rs",
                "src/base.rs",
                "src/stable.rs",
            ] {
                let bytes = std::fs::read(root.join(rel)).unwrap();
                let metadata = greppy_discover::stable_metadata(
                    &std::fs::symlink_metadata(root.join(rel)).unwrap(),
                );
                base.upsert_file_state(&greppy_store::FileState {
                    project: "p".into(),
                    rel_path: rel.into(),
                    language: "Rust".into(),
                    sha256: greppy_store::file_state::sha256_hex(&bytes),
                    mtime_ns: metadata.mtime_ns.unwrap_or_default(),
                    size: metadata.size as i64,
                    parser_version: "fixture".into(),
                    extractor_version: "fixture".into(),
                    last_indexed_generation: 7,
                })
                .unwrap();
            }
            base.insert_raw_edges(&[
                greppy_store::NewRawEdge {
                    project: "p".into(),
                    file_path: "src/alias_chain/mod.rs".into(),
                    source_qname: "src/alias_chain/mod.rs::__file__".into(),
                    target_qname: "src/alias_chain/mod.rs::Import::sub::target".into(),
                    edge_type: "IMPORTS".into(),
                    properties: serde_json::json!({
                        "imported_name": "target",
                        "imported_items": [{
                            "path": "sub::target",
                            "imported_name": "outer",
                            "original_name": "target",
                            "glob": false
                        }]
                    }),
                },
                greppy_store::NewRawEdge {
                    project: "p".into(),
                    file_path: "src/stable.rs".into(),
                    source_qname: "src/stable.rs::Function::stable_caller".into(),
                    target_qname: "src/alias_chain/sub.rs::Function::target".into(),
                    edge_type: "CALLS".into(),
                    properties: serde_json::json!({"callee_name": "target"}),
                },
                greppy_store::NewRawEdge {
                    project: "p".into(),
                    file_path: "src/base.rs".into(),
                    source_qname: "src/base.rs::Function::base_caller".into(),
                    target_qname: "src/alias_chain/sub.rs::Function::target".into(),
                    edge_type: "CALLS".into(),
                    properties: serde_json::json!({"callee_name": "target"}),
                },
            ])
            .unwrap();
        }
        timing("base-store-populated");
        let base_identity = base_identity_parts(&root, &base_commit).unwrap();
        let base_layout = BaseStoreLayout::new(scratch.path(), &base_identity).unwrap();
        let summary_dir = scratch.path().join("base-summary-cache");
        let summary_path = {
            let summary = greppy_store::SummaryCache::open(&summary_dir).unwrap();
            drop(summary);
            summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE)
        };
        let _base_lease = base_layout.acquire_builder(true).unwrap().unwrap();
        base_layout
            .publish_graph_with_summary(base_identity, &staged_base_path, &summary_path)
            .unwrap();
        timing("base-published");
        let base_path = base_layout.graph.clone();
        let caller_rel_path = "src/caller.rs";
        let caller_metadata = greppy_discover::stable_metadata(
            &std::fs::symlink_metadata(root.join(caller_rel_path)).unwrap(),
        );
        let caller_sha256 = greppy_store::file_state::sha256_hex(
            &std::fs::read(root.join(caller_rel_path)).unwrap(),
        );
        // Match a real publication's repository fingerprint. Leaving these
        // fields empty makes the first query classify this synthetic Delta as
        // stale and try to launch the CLI through the libtest executable.
        // The dirty file state remains generation 7, so the query still has
        // to exercise the bounded persisted repair below.
        let fixture_fingerprint = greppy_core::GitFingerprint::capture(&root);
        {
            let mut delta = greppy_store::Store::open(&delta_path).unwrap();
            delta
                .upsert_project(&greppy_store::Project {
                    name: "p".into(),
                    indexed_at: "2026-09-27T00:00:00Z".into(),
                    root_path: root.to_string_lossy().into_owned(),
                })
                .unwrap();
            delta
                .upsert_file_state(&greppy_store::FileState {
                    project: "p".into(),
                    rel_path: caller_rel_path.into(),
                    language: "Rust".into(),
                    sha256: caller_sha256,
                    mtime_ns: caller_metadata.mtime_ns.unwrap_or_default(),
                    size: caller_metadata.size as i64,
                    parser_version: "fixture".into(),
                    extractor_version: "fixture".into(),
                    last_indexed_generation: 7,
                })
                .unwrap();
            delta
                .upsert_file_identity(
                    "p",
                    caller_rel_path,
                    greppy_store::FileIdentity {
                        ctime_ns: caller_metadata.ctime_ns,
                        file_id: caller_metadata.file_id,
                    },
                )
                .unwrap();
            delta
                .upsert_workspace_state(&greppy_store::WorkspaceState {
                    root_path: root.to_string_lossy().into_owned(),
                    git_dir: fixture_fingerprint
                        .git_dir
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    git_common_dir: fixture_fingerprint
                        .git_common_dir
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    head_oid: fixture_fingerprint.head_oid.clone(),
                    index_signature: fixture_fingerprint.index_signature.clone(),
                    schema_version: delta.schema_version().unwrap(),
                    indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
                    graph_generation: 7,
                    updated_at: "2026-09-27T00:00:00Z".into(),
                })
                .unwrap();
            delta
                .conn()
                .execute(
                    "INSERT OR REPLACE INTO main.schema_meta (key, value) VALUES (?1, ?2)",
                    [
                        "greppy.rust_caller_edges_repair.v2",
                        RUST_CALLER_EDGES_REPAIR_COMPLETE,
                    ],
                )
                .unwrap();
            delta
                .insert_node(&greppy_store::NewNode {
                    project: "p".into(),
                    label: "Module".into(),
                    name: "caller".into(),
                    qualified_name: "src/caller.rs::__file__".into(),
                    file_path: "src/caller.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            delta
                .insert_node(&greppy_store::NewNode {
                    project: "p".into(),
                    label: "Function".into(),
                    name: "caller".into(),
                    qualified_name: "src/caller.rs::Function::caller".into(),
                    file_path: "src/caller.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    properties: serde_json::json!({}),
                })
                .unwrap();
            delta
                .insert_raw_edges(&[
                    greppy_store::NewRawEdge {
                        project: "p".into(),
                        file_path: "src/caller.rs".into(),
                        source_qname: "src/caller.rs::__file__".into(),
                        target_qname: "src/caller.rs::Import::crate::alias_chain::outer".into(),
                        edge_type: "IMPORTS".into(),
                        properties: serde_json::json!({
                            "imported_name": "outer",
                            "imported_items": [{
                                "path": "crate::alias_chain::outer",
                                "imported_name": "outer",
                                "original_name": "outer",
                                "glob": false
                            }]
                        }),
                    },
                    greppy_store::NewRawEdge {
                        project: "p".into(),
                        file_path: "src/caller.rs".into(),
                        source_qname: "src/caller.rs::Function::caller".into(),
                        target_qname: "src/caller.rs::__ref__::outer".into(),
                        edge_type: "USAGE".into(),
                        properties: serde_json::json!({"ref_name": "outer"}),
                    },
                ])
                .unwrap();
            delta
                .upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                    project: "p".into(),
                    model_id: "fixture".into(),
                    prompt_version: "fixture".into(),
                    task: "code".into(),
                    node_id: None,
                    chunk_idx: 0,
                    qualified_name: "src/caller.rs::Function::caller".into(),
                    file_path: "src/caller.rs".into(),
                    start_line: 1,
                    end_line: 1,
                    content_sha256: "a".repeat(64),
                    graph_generation: 7,
                    vector: vec![1.0, 0.0],
                })
                .unwrap();
        }

        timing("delta-populated");
        let visibility =
            greppy_store::VisibilityIndex::new(["src/caller.rs".to_string()], Vec::<String>::new())
                .unwrap();
        {
            let delta = greppy_store::Store::open(&delta_path).unwrap();
            persist_visibility(&delta, &visibility, &base_commit).unwrap();
        }
        std::env::set_var(ENV_MODE, MODE_OVERLAY);
        std::env::set_var(ENV_BASE_PATH, &base_path);
        std::env::set_var(ENV_BASE_COMMIT, &base_commit);
        let legacy =
            greppy_store::Store::open_overlay(&base_path, &delta_path, &visibility).unwrap();
        assert_eq!(
            legacy
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM main.overlay_edges WHERE project = 'p'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0,
            "the preserved pre-repair Delta starts without the corrected edge"
        );
        let legacy_target = legacy
            .get_node_by_qname("p", "src/alias_chain/sub.rs::Function::target")
            .unwrap()
            .unwrap();
        assert!(legacy
            .incoming_edges(legacy_target.id, Some("USAGE"), 10)
            .unwrap()
            .is_empty());
        let legacy_base_caller = legacy
            .get_node_by_qname("p", "src/base.rs::Function::base_caller")
            .unwrap()
            .unwrap();
        assert!(
            legacy
                .incoming_edges(legacy_target.id, Some("CALLS"), 10)
                .unwrap()
                .iter()
                .all(|edge| edge.source_id != legacy_base_caller.id),
            "the missing Base caller has no stale logical edge before repair"
        );
        let root_string = root.to_string_lossy().into_owned();
        let freshness_proof = crate::nav_freshness_json(&legacy, Some(&root_string), "p");
        assert!(
            freshness_proof["fresh"] == true
                && freshness_proof["source"] == "verified_store_cow_overlay",
            "persisted repair fixture must satisfy the real Store-CoW freshness gate before the query; otherwise the unit-test executable would be selected as a background CLI: {freshness_proof:?}"
        );
        drop(legacy);
        let vector_before: Vec<u8> =
            greppy_store::Store::open_with(&delta_path, greppy_store::OpenOptions::read_only())
                .unwrap()
                .conn()
                .query_row(
                    "SELECT vector FROM main.vector_embeddings WHERE project = 'p'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
        timing("freshness-proven");
        let held = greppy_freshness::try_acquire(&delta_path).unwrap();
        let start = std::sync::Arc::new(std::sync::Barrier::new(3));
        let first_start = std::sync::Arc::clone(&start);
        let first_root = root_string.clone();
        let first = std::thread::spawn(move || {
            first_start.wait();
            assert_persisted_v7_delta_query_is_correct(&first_root)
        });
        let second_start = std::sync::Arc::clone(&start);
        let second_root = root_string.clone();
        let second = std::thread::spawn(move || {
            second_start.wait();
            assert_persisted_v7_delta_query_is_correct(&second_root)
        });
        start.wait();
        // Exercise the normal freshness wait budget rather than the old
        // repair-local cap: a legitimate graph-only repair may exceed two
        // seconds while still being a live publication.
        timing("contending-queries-started");
        std::thread::sleep(std::time::Duration::from_millis(2_500));
        timing("fixture-sleep-ended");
        drop(held);
        timing("fixture-lock-released");
        first
            .join()
            .unwrap_or_else(|_| Err("first concurrent query panicked".into()))
            .unwrap();
        second
            .join()
            .unwrap_or_else(|_| Err("second concurrent query panicked".into()))
            .unwrap();
        let repaired = crate::freshness::open_default_store(Some(&root_string)).unwrap();
        let target = repaired
            .get_node_by_qname("p", "src/alias_chain/sub.rs::Function::target")
            .unwrap()
            .unwrap();
        let caller = repaired
            .get_node_by_qname("p", "src/caller.rs::Function::caller")
            .unwrap()
            .unwrap();
        assert!(repaired
            .incoming_edges(target.id, Some("USAGE"), 10)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == caller.id));
        let base_caller = repaired
            .get_node_by_qname("p", "src/base.rs::Function::base_caller")
            .unwrap()
            .unwrap();
        assert!(repaired
            .incoming_edges(target.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == base_caller.id));
        let stable_caller = repaired
            .get_node_by_qname("p", "src/stable.rs::Function::stable_caller")
            .unwrap()
            .unwrap();
        assert!(repaired
            .incoming_edges(target.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == stable_caller.id));
        assert_eq!(
            repaired
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM main.overlay_edges
                     WHERE project = 'p'
                       AND source_qualified_name = 'src/base.rs::Function::base_caller'
                       AND json_extract(properties, '$.greppy_base_repair_v2') = 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1,
            "only the missing Base relation becomes a repair overlay"
        );
        assert_eq!(
            repaired
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM main.overlay_edges
                     WHERE project = 'p'
                       AND source_qualified_name = 'src/stable.rs::Function::stable_caller'
                       AND json_extract(properties, '$.greppy_base_repair_v2') = 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0,
            "an unchanged Base relation is not copied into the repair set"
        );
        assert_eq!(
            repaired
                .get_workspace_state(&root_string)
                .unwrap()
                .unwrap()
                .graph_generation,
            7,
            "edge repair must not publish a new graph generation"
        );
        let vector_after: Vec<u8> = repaired
            .conn()
            .query_row(
                "SELECT vector FROM main.vector_embeddings WHERE project = 'p'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(vector_after, vector_before);
        let overlay_relations = |store: &greppy_store::Store| {
            let mut statement = store
                .conn()
                .prepare(
                    "SELECT source_qualified_name, target_qualified_name, edge_type,
                            COALESCE(json_extract(properties, '$.greppy_base_repair_v2'), 0)
                     FROM main.overlay_edges WHERE project = 'p'
                     ORDER BY source_qualified_name, target_qualified_name, edge_type",
                )
                .unwrap();
            let relations = statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            relations
        };
        // The one-shot composed rebuild shadows an existing Base relation too.
        // Store visibility suppresses the matching Base row, so this is one
        // visible relation, not two. Only missing Base edges carry repair markers.
        // Exact reexport resolution also retains the caller's import of outer.
        let expected_relations = [
            ("src/alias_chain/mod.rs::__file__", "IMPORTS", 1),
            ("src/base.rs::Function::base_caller", "CALLS", 1),
            ("src/caller.rs::Function::caller", "USAGE", 0),
            ("src/caller.rs::__file__", "IMPORTS", 0),
            ("src/stable.rs::Function::stable_caller", "CALLS", 0),
        ]
        .into_iter()
        .map(|(source, kind, repaired)| {
            (
                source.to_string(),
                "src/alias_chain/sub.rs::Function::target".to_string(),
                kind.to_string(),
                repaired,
            )
        })
        .collect::<Vec<_>>();
        assert_eq!(overlay_relations(&repaired), expected_relations);
        let vector_count = repaired
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM main.vector_embeddings WHERE project = 'p'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(vector_count, 1);
        drop(repaired);

        let warm = crate::freshness::open_default_store(Some(&root_string)).unwrap();
        assert_eq!(
            warm.conn()
                .query_row(
                    "SELECT value FROM main.schema_meta WHERE key = ?1",
                    [RUST_CALLER_EDGES_REPAIR_META_KEY],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            RUST_CALLER_EDGES_REPAIR_COMPLETE
        );
        assert_eq!(
            warm.get_workspace_state(&root_string)
                .unwrap()
                .unwrap()
                .graph_generation,
            7
        );
        drop(warm);

        // A later ordinary dirty-file publication rebuilds Delta-owned raw
        // edges. The repaired Base-derived overlay edge must remain visible;
        // it cannot depend on rescanning Base raw edges on every query.
        std::fs::write(
            root.join(caller_rel_path),
            "use crate::alias_chain::outer;\npub fn caller() { outer(); }\n// second dirty Delta\n",
        )
        .unwrap();
        // A production drift query launches `<current greppy> index ...` with
        // structural-first-use set. This unit test runs inside the libtest
        // executable, so spawning current_exe would feed CLI arguments to the
        // test harness. Dispatch the same structural index path in-process;
        // embeddings remain deferred and the vector-preservation assertion
        // below continues to cover the one-shot repair contract. Use an
        // explicit CLI-sized stack rather than the platform's libtest stack.
        std::env::set_var(crate::ENV_STRUCTURAL_FIRST_USE, "1");
        let index_root = root_string.clone();
        let index_thread = match std::thread::Builder::new()
            .name("persisted-repair-structural-index".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                crate::dispatch(
                    crate::Cli::try_parse_from([
                        "greppy",
                        "index",
                        &index_root,
                        "--root",
                        &index_root,
                    ])
                    .unwrap(),
                )
            }) {
            Ok(thread) => thread,
            Err(error) => {
                std::env::remove_var(crate::ENV_STRUCTURAL_FIRST_USE);
                panic!("cannot spawn structural index test thread: {error}");
            }
        };
        let index_result = index_thread.join();
        std::env::remove_var(crate::ENV_STRUCTURAL_FIRST_USE);
        let index_code = index_result
            .unwrap_or_else(|_| panic!("structural index test thread panicked"))
            .unwrap();
        assert_eq!(index_code, 0, "dirty structural publication should succeed");
        let code = crate::dispatch(
            crate::Cli::try_parse_from(["greppy", "--root", &root_string, "who-calls", "target"])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(code, 0, "dirty publication should succeed");
        let after_dirty = crate::freshness::open_default_store(Some(&root_string)).unwrap();
        let target_after_dirty = after_dirty
            .get_node_by_qname("p", "src/alias_chain/sub.rs::Function::target")
            .unwrap()
            .unwrap();
        let base_caller_after_dirty = after_dirty
            .get_node_by_qname("p", "src/base.rs::Function::base_caller")
            .unwrap()
            .unwrap();
        assert!(after_dirty
            .incoming_edges(target_after_dirty.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == base_caller_after_dirty.id));
        let stable_caller_after_dirty = after_dirty
            .get_node_by_qname("p", "src/stable.rs::Function::stable_caller")
            .unwrap()
            .unwrap();
        assert!(after_dirty
            .incoming_edges(target_after_dirty.id, Some("CALLS"), 10)
            .unwrap()
            .iter()
            .any(|edge| edge.source_id == stable_caller_after_dirty.id));
        let later_relations = overlay_relations(&after_dirty);
        assert!(
            later_relations
                .iter()
                .all(|row| row.0 != "src/stable.rs::Function::stable_caller"),
            "ordinary bounded publication must prune the temporary Base shadow"
        );
        assert_eq!(
            later_relations
                .iter()
                .filter(|row| row.3 == 1)
                .collect::<Vec<_>>(),
            expected_relations
                .iter()
                .filter(|row| row.3 == 1)
                .collect::<Vec<_>>(),
            "both missing Base relations must survive ordinary Delta publication"
        );
        let caller_after_dirty = after_dirty
            .get_node_by_qname("p", "src/caller.rs::Function::caller")
            .unwrap()
            .unwrap();
        assert!(
            after_dirty
                .incoming_edges(target_after_dirty.id, None, 20)
                .unwrap()
                .iter()
                .any(|edge| edge.source_id == caller_after_dirty.id
                    && matches!(edge.edge_type.as_str(), "CALLS" | "USAGE")),
            "the real parser must republish the dirty caller relation"
        );
        // The edge-only compatibility repair above must preserve vectors.
        // This later phase actually changes caller.rs, so its stale embedding
        // must be invalidated by ordinary file reindexing, not carried forward.
        let stale_caller_vectors: i64 = after_dirty
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM main.vector_embeddings WHERE project = 'p' AND file_path = 'src/caller.rs'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stale_caller_vectors, 0);
    }

    #[test]
    fn v7_base_seed_copies_verified_v6_graph_and_summary() {
        let data_root = tempfile::tempdir().unwrap();
        let sources = tempfile::tempdir().unwrap();
        let graph = sources.path().join("graph.db");
        let summary_dir = sources.path().join("summary");
        let summary = {
            let cache = greppy_store::SummaryCache::open(&summary_dir).unwrap();
            drop(cache);
            summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE)
        };
        let root = "/old/base/root";
        {
            let mut store = greppy_store::Store::open(&graph).unwrap();
            store
                .upsert_project(&greppy_store::Project {
                    name: "fixture".into(),
                    indexed_at: "2026-09-27T00:00:00Z".into(),
                    root_path: root.into(),
                })
                .unwrap();
            store
                .upsert_workspace_state(&greppy_store::WorkspaceState {
                    root_path: root.into(),
                    git_dir: None,
                    git_common_dir: None,
                    head_oid: None,
                    index_signature: None,
                    schema_version: store.schema_version().unwrap(),
                    indexer_version: "greppy-indexer-v6".into(),
                    graph_generation: 1,
                    updated_at: "2026-09-27T00:00:00Z".into(),
                })
                .unwrap();
        }
        let previous_identity = BaseStoreIdentity {
            format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
            canonical_repository_identity: "fixture-repository".into(),
            git_object_format: "sha1".into(),
            base_tree_oid: "1111111111111111111111111111111111111111".into(),
            store_schema_version: greppy_store::migrate::CURRENT_VERSION,
            indexer_version: "greppy-indexer-v6".into(),
            parser_and_extractor_versions: "fixture-parser".into(),
            summary_model_and_prompt_version: "fixture-summary".into(),
            embedding_model: "fixture-embedding".into(),
            embedding_prompt_version: "fixture-prompt".into(),
            embedding_dimensions: 2,
            embedding_encoding: "f32+i8-v1".into(),
        };
        let previous_layout = BaseStoreLayout::new(data_root.path(), &previous_identity).unwrap();
        previous_layout
            .publish_graph_with_summary(previous_identity.clone(), &graph, &summary)
            .unwrap();
        let previous_summary_hash = greppy_store::file_state::sha256_hex(
            &std::fs::read(&previous_layout.summary_cache).unwrap(),
        );

        let mut current_identity = previous_identity;
        current_identity.indexer_version = "greppy-indexer-v7".into();
        assert!(
            has_verified_previous_indexer_base_for_identity(data_root.path(), &current_identity)
                .unwrap(),
            "a verified v6 Base must force structural first-use migration"
        );
        let migrated_root = data_root.path().join("migrated-worktree");
        std::fs::create_dir_all(&migrated_root).unwrap();
        let migrated_root = migrated_root.canonicalize().unwrap();
        let staged_graph = data_root.path().join("staging/workspaces/fixture/graph.db");
        let staged_summary = seed_previous_indexer_base(
            data_root.path(),
            &current_identity,
            &migrated_root,
            &staged_graph,
        )
        .unwrap()
        .expect("verified v6 Base should seed v7 staging");

        assert_eq!(
            greppy_store::file_state::sha256_hex(&std::fs::read(&staged_summary).unwrap()),
            previous_summary_hash
        );
        let migrated =
            greppy_store::Store::open_with(&staged_graph, greppy_store::OpenOptions::read_only())
                .unwrap();
        assert_eq!(
            migrated.list_projects().unwrap()[0].root_path,
            migrated_root.to_string_lossy()
        );
        assert_eq!(
            migrated.list_workspace_states().unwrap()[0].root_path,
            migrated_root.to_string_lossy()
        );
        assert_eq!(
            greppy_store::file_state::sha256_hex(
                &std::fs::read(&previous_layout.summary_cache).unwrap()
            ),
            previous_summary_hash,
            "published v6 summary cache remains immutable"
        );
    }

    #[test]
    fn deferred_embedding_receipt_is_bound_to_generation_and_model() {
        let expected = "7|model-a";

        assert!(base_embedding_receipt_valid(None, Some(expected), expected));
        assert!(!base_embedding_receipt_valid(
            None,
            Some("8|model-a"),
            expected
        ));
        assert!(!base_embedding_receipt_valid(
            None,
            Some("7|model-b"),
            expected
        ));
    }

    #[test]
    fn worktree_list_prefers_nul_porcelain_and_preserves_newlines() {
        let mut common_dir_called = false;
        let paths = compatible_worktree_paths(
            || {
                Ok(b"worktree /repo with spaces\0HEAD deadbeef\0\0worktree /repo\nwith-newline\0bare\0\0".to_vec())
            },
            || {
                common_dir_called = true;
                Ok(PathBuf::from("/unused/.git"))
            },
        )
        .unwrap();

        assert!(!common_dir_called);
        assert_eq!(
            paths,
            [
                PathBuf::from("/repo with spaces"),
                PathBuf::from("/repo\nwith-newline")
            ]
        );
    }

    #[test]
    fn worktree_list_falls_back_to_common_dir_without_parsing_legacy_output() {
        let paths = compatible_worktree_paths(
            || Err(Error::Invalid("unknown switch `z'".into())),
            || Ok(PathBuf::from("/primary path\nwith-newline/.git")),
        )
        .unwrap();

        assert_eq!(paths, [PathBuf::from("/primary path\nwith-newline")]);
    }

    #[test]
    fn worktree_list_reports_modern_and_common_dir_failures() {
        let error = compatible_worktree_paths(
            || Err(Error::Invalid("unknown switch `z'".into())),
            || Err(Error::Invalid("not a git repository".into())),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("unknown switch `z'"), "{error}");
        assert!(error.contains("not a git repository"), "{error}");
    }

    #[test]
    fn worktree_list_rejects_bare_common_dir_fallback() {
        let error = compatible_worktree_paths(
            || Err(Error::Invalid("unknown switch `z'".into())),
            || Ok(PathBuf::from("/repositories/project.git")),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("does not identify a primary checkout"),
            "{error}"
        );
    }

    #[test]
    fn temporary_base_checkout_uses_tmpdir_and_cleans_up() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let scratch = tempfile::tempdir().unwrap();
        let _restore = TmpdirRestore::set(scratch.path());
        let repo = fixture();
        let commit = git(repo.path(), &["rev-parse", "HEAD"]);

        let checkout = TemporaryBaseWorktree::create(repo.path(), &commit).unwrap();
        let checkout_parent = checkout._parent.path().to_path_buf();
        assert_eq!(checkout_parent.parent(), Some(scratch.path()));
        assert!(checkout.path().join(".git").is_file());

        drop(checkout);
        assert!(!checkout_parent.exists());
        assert!(git(repo.path(), &["worktree", "list", "--porcelain"])
            .lines()
            .all(|line| !line.contains("greppy-linked-base-checkout-")));
    }

    #[test]
    fn temporary_base_checkout_refuses_missing_configured_tmpdir() {
        let _env = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let scratch_parent = tempfile::tempdir().unwrap();
        let missing = scratch_parent.path().join("missing-scratch");
        let repo = fixture();
        let commit = git(repo.path(), &["rev-parse", "HEAD"]);
        let _restore = TmpdirRestore::set(&missing);

        let error = match TemporaryBaseWorktree::create(repo.path(), &commit) {
            Ok(_) => panic!("missing TMPDIR unexpectedly accepted"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("temporary Base checkout directory"),
            "{message}"
        );
        assert!(
            message.contains(missing.to_string_lossy().as_ref()),
            "{message}"
        );
        assert!(
            !missing.exists(),
            "invalid TMPDIR must not be created or bypassed"
        );
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
    fn concurrent_base_builder_wait_honors_caller_deadline() {
        let repo = fixture();
        let commit = git(repo.path(), &["rev-parse", "HEAD"]);
        let identity = base_identity_parts(repo.path(), &commit).unwrap();
        let identity_hash = identity.hash().unwrap();
        let data_root = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(data_root.path(), &identity).unwrap();
        let held = layout.acquire_builder(true).unwrap().unwrap();
        let progress_path = data_root.path().join("index.job");
        crate::start_background_job_record(
            &progress_path,
            &serde_json::json!({
                "schema_version": crate::BACKGROUND_JOB_SCHEMA_VERSION,
                "kind": "index",
                "pid": std::process::id(),
                "target_generation": 1,
                "started_at_unix_secs": 1,
                "updated_at_unix_secs": 1,
                "state": "preparing_base_checkout"
            }),
        )
        .unwrap();

        let started = std::time::Instant::now();
        let deadline = started + std::time::Duration::from_millis(20);
        let error = match acquire_base_builder(
            &layout,
            &identity_hash,
            Some(&progress_path),
            Some(deadline),
            None,
        ) {
            Ok(_) => panic!("second Base builder unexpectedly acquired the held lease"),
            Err(error) => error,
        };

        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        let message = error.to_string();
        assert!(message.contains("deadline reached while waiting for immutable Base"));
        assert!(message.contains(&identity_hash));
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
        drop(held);
        let free_error =
            match acquire_base_builder(&layout, &identity_hash, None, Some(deadline), None) {
                Ok(_) => panic!("expired caller acquired a free Base builder lease"),
                Err(error) => error,
            };
        assert!(free_error
            .to_string()
            .contains("deadline reached while waiting for immutable Base"));
    }

    #[test]
    fn concurrent_base_builder_wait_honors_cancellation() {
        let repo = fixture();
        let commit = git(repo.path(), &["rev-parse", "HEAD"]);
        let identity = base_identity_parts(repo.path(), &commit).unwrap();
        let identity_hash = identity.hash().unwrap();
        let data_root = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(data_root.path(), &identity).unwrap();
        let held = layout.acquire_builder(true).unwrap().unwrap();
        let cancel = std::sync::atomic::AtomicBool::new(true);

        let error = match acquire_base_builder(&layout, &identity_hash, None, None, Some(&cancel)) {
            Ok(_) => panic!("cancelled consumer acquired the held Base builder lease"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains("cancelled while waiting for immutable Base"));
        assert!(message.contains(&identity_hash));
        drop(held);
        let free_error =
            match acquire_base_builder(&layout, &identity_hash, None, None, Some(&cancel)) {
                Ok(_) => panic!("cancelled consumer acquired a free Base builder lease"),
                Err(error) => error,
            };
        assert!(free_error
            .to_string()
            .contains("cancelled while waiting for immutable Base"));
    }

    #[test]
    fn concurrent_base_consumer_waits_for_owner_publication_lock() {
        let repo = fixture();
        let commit = git(repo.path(), &["rev-parse", "HEAD"]);
        let identity = base_identity_parts(repo.path(), &commit).unwrap();
        let identity_hash = identity.hash().unwrap();
        let data_root = tempfile::tempdir().unwrap();
        let layout = BaseStoreLayout::new(data_root.path(), &identity).unwrap();
        let owner = layout.acquire_builder(true).unwrap().unwrap();
        let publication = layout.directory.join("test-publication-complete");
        let (sent, received) = std::sync::mpsc::channel();
        let waiting_layout = layout.clone();
        let waiting_identity = identity_hash.clone();
        let waiting_publication = publication.clone();
        let waiter = std::thread::spawn(move || {
            let lease = acquire_base_builder(&waiting_layout, &waiting_identity, None, None, None)
                .expect("consumer must acquire the lifecycle lease after its owner publishes");
            sent.send((lease, waiting_publication.is_file())).unwrap();
        });

        assert!(received
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        std::fs::create_dir_all(&layout.directory).unwrap();
        std::fs::write(&publication, b"published").unwrap();
        drop(owner);
        let (consumer, observed_publication) = received
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("consumer did not resume after Base owner released publication lock");
        assert!(observed_publication);
        drop(consumer);
        waiter.join().unwrap();
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
            &std::collections::HashMap::new(),
        )
        .unwrap());
    }

    #[test]
    fn sparse_checkout_delta_freshness_uses_the_staged_blob() {
        let repo = fixture();
        let base = git(repo.path(), &["rev-parse", "HEAD"]);
        std::fs::create_dir_all(repo.path().join("docs")).unwrap();
        std::fs::create_dir_all(repo.path().join(".github/workflows")).unwrap();
        std::fs::write(repo.path().join("docs/added.rs"), "fn added() {}\n").unwrap();
        std::fs::write(repo.path().join("docs/[literal].rs"), "fn literal() {}\n").unwrap();
        std::fs::write(repo.path().join(".github/workflows/ci.yml"), "name: CI\n").unwrap();
        git(
            repo.path(),
            &[
                "add",
                "docs/added.rs",
                "docs/[literal].rs",
                ".github/workflows/ci.yml",
            ],
        );
        git(repo.path(), &["commit", "-q", "-m", "add sparse files"]);

        let mut store = greppy_store::Store::open_memory().unwrap();
        let options = greppy_indexer::IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from([
                "docs/added.rs".to_string(),
                "docs/[literal].rs".to_string(),
                ".github/workflows/ci.yml".to_string(),
            ])),
            ..greppy_indexer::IndexOptions::default()
        };
        greppy_indexer::index_with_options(&mut store, repo.path(), "p", &options).unwrap();
        assert!(store
            .get_file_state("p", "docs/added.rs")
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .get_index_skip("p", ".github/workflows/ci.yml")
                .unwrap()
                .unwrap()
                .reason,
            "discovery_filtered"
        );
        assert!(store
            .get_file_state("p", "docs/[literal].rs")
            .unwrap()
            .is_some());

        git(repo.path(), &["sparse-checkout", "init", "--cone"]);
        git(repo.path(), &["sparse-checkout", "set", "src"]);
        assert!(!repo.path().join("docs/added.rs").exists());
        assert!(!repo.path().join("docs/[literal].rs").exists());
        assert!(!repo.path().join(".github/workflows/ci.yml").exists());
        let visibility = visibility_against(repo.path(), &base).unwrap();
        assert!(visibility.is_dirty_path("docs/added.rs"));
        assert!(visibility.is_dirty_path("docs/[literal].rs"));
        assert!(visibility.is_dirty_path(".github/workflows/ci.yml"));
        let sparse_paths = std::collections::BTreeSet::from([
            "docs/added.rs".to_string(),
            "docs/[literal].rs".to_string(),
            ".github/workflows/ci.yml".to_string(),
        ]);
        let sparse_blobs = persisted_sparse_delta_blobs(repo.path(), &sparse_paths).unwrap();
        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "docs/added.rs",
            &store.list_file_identities("p").unwrap(),
            &sparse_blobs,
        )
        .unwrap());
        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            ".github/workflows/ci.yml",
            &store.list_file_identities("p").unwrap(),
            &sparse_blobs,
        )
        .unwrap());
        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "docs/[literal].rs",
            &store.list_file_identities("p").unwrap(),
            &sparse_blobs,
        )
        .unwrap());

        let replacement = repo.path().join("replacement.rs");
        std::fs::write(&replacement, "fn replacement() {}\n").unwrap();
        let replacement_oid = git(
            repo.path(),
            &["hash-object", "-w", replacement.to_str().unwrap()],
        );
        let cache_entry = format!("100644,{replacement_oid},docs/added.rs");
        git(repo.path(), &["update-index", "--cacheinfo", &cache_entry]);
        git(
            repo.path(),
            &["update-index", "--skip-worktree", "docs/added.rs"],
        );
        let sparse_blobs = persisted_sparse_delta_blobs(repo.path(), &sparse_paths).unwrap();
        assert!(!persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "docs/added.rs",
            &store.list_file_identities("p").unwrap(),
            &sparse_blobs,
        )
        .unwrap());
        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "docs/[literal].rs",
            &store.list_file_identities("p").unwrap(),
            &sparse_blobs,
        )
        .unwrap());

        git(
            repo.path(),
            &["update-index", "--force-remove", "docs/added.rs"],
        );
        let sparse_blobs = persisted_sparse_delta_blobs(repo.path(), &sparse_paths).unwrap();
        assert!(!persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "docs/added.rs",
            &store.list_file_identities("p").unwrap(),
            &sparse_blobs,
        )
        .unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn tracked_symlink_skip_uses_symlink_identity_without_following_target() {
        use std::os::unix::fs::symlink;

        let repo = fixture();
        std::fs::write(repo.path().join("AGENTS.md"), "small target\n").unwrap();
        symlink("AGENTS.md", repo.path().join("CLAUDE.md")).unwrap();
        git(repo.path(), &["add", "AGENTS.md", "CLAUDE.md"]);
        git(repo.path(), &["commit", "-q", "-m", "add tracked symlink"]);

        let mut store = greppy_store::Store::open_memory().unwrap();
        let options = greppy_indexer::IndexOptions {
            only_paths: Some(std::collections::BTreeSet::from(["CLAUDE.md".to_string()])),
            ..greppy_indexer::IndexOptions::default()
        };
        greppy_indexer::index_with_options(&mut store, repo.path(), "p", &options).unwrap();
        assert!(store.get_index_skip("p", "CLAUDE.md").unwrap().is_some());
        assert!(store.get_file_state("p", "CLAUDE.md").unwrap().is_none());

        // Target content is not the identity of the tracked link.
        std::fs::write(repo.path().join("AGENTS.md"), "different target content\n").unwrap();

        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "CLAUDE.md",
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        )
        .unwrap());

        std::fs::remove_file(repo.path().join("CLAUDE.md")).unwrap();
        symlink("MISSING.md", repo.path().join("CLAUDE.md")).unwrap();
        assert!(!persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "CLAUDE.md",
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        )
        .unwrap());
        // A broken link is also a valid filtered entry after refresh.
        greppy_indexer::index_with_options(&mut store, repo.path(), "p", &options).unwrap();
        assert!(store.get_file_state("p", "CLAUDE.md").unwrap().is_none());
        assert!(persisted_delta_path_matches(
            repo.path(),
            &store,
            "p",
            "CLAUDE.md",
            &std::collections::HashMap::new(),
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
