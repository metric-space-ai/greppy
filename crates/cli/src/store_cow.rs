//! Store-CoW lifecycle shared by `greppy -p`, index warming, and query opens.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use greppy_core::error::{Error, Result};
use greppy_store::{BaseStoreIdentity, BaseStoreLayout, VisibilityIndex};
use sha2::{Digest, Sha256};

pub(crate) const ENV_MODE: &str = "GREPPY_AGENT_STORE_MODE";
pub(crate) const ENV_BASE_PATH: &str = "GREPPY_AGENT_BASE_STORE";
pub(crate) const ENV_BASE_COMMIT: &str = "GREPPY_AGENT_BASE_COMMIT";
pub(crate) const ENV_BASE_REUSED: &str = "GREPPY_AGENT_BASE_REUSED";
pub(crate) const ENV_FALLBACK_REASON: &str = "GREPPY_AGENT_STORE_FALLBACK_REASON";
pub(crate) const MODE_OVERLAY: &str = "overlay";
pub(crate) const MODE_PRIVATE: &str = "private";
const VISIBILITY_META_KEY: &str = "store_cow.visibility.v1";

#[derive(Debug, Clone)]
pub(crate) struct OverlaySpec {
    pub base_path: PathBuf,
    pub visibility: VisibilityIndex,
}

#[derive(Debug)]
pub(crate) struct PreparedBase {
    pub graph_path: PathBuf,
    pub identity_hash: String,
    pub reused: bool,
    _reader_lease: greppy_store::BaseReaderLease,
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
    let Some((base_path, base_commit)) = overlay_environment()? else {
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

pub(crate) fn overlay_environment() -> Result<Option<(PathBuf, String)>> {
    if std::env::var(ENV_MODE).ok().as_deref() != Some(MODE_OVERLAY) {
        return Ok(None);
    }
    let base_path = std::env::var_os(ENV_BASE_PATH)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Invalid(format!("{ENV_MODE}=overlay without {ENV_BASE_PATH}")))?;
    if !base_path.is_file() {
        return Err(Error::Invalid(format!(
            "configured immutable Base Store is missing: {}",
            base_path.display()
        )));
    }
    let base_commit = std::env::var(ENV_BASE_COMMIT)
        .map_err(|_| Error::Invalid(format!("{ENV_MODE}=overlay without {ENV_BASE_COMMIT}")))?;
    Ok(Some((base_path, base_commit)))
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
    let configured_mode = std::env::var(ENV_MODE).ok();
    let fallback_reason = std::env::var(ENV_FALLBACK_REASON).ok();
    let base_commit = std::env::var(ENV_BASE_COMMIT).ok();
    let base_reused = std::env::var(ENV_BASE_REUSED)
        .ok()
        .as_deref()
        .map(|value| value == "1");

    let mut base_path = None;
    let mut base_identity = None;
    let mut base_complete = None;
    let mut dirty_paths = None;
    let mut deleted_paths = None;
    let mut delta_identity = None;
    let mut error = None;
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

pub(crate) fn prepare_base_store(
    workspace: &greppy_agent::workspace::AgentWorkspace,
    shared_data_root: &Path,
) -> Result<PreparedBase> {
    let identity = base_identity(workspace)?;
    let layout = BaseStoreLayout::new(shared_data_root, &identity)
        .map_err(|error| Error::io("construct Base Store layout", error))?;
    if let Ok(manifest) = layout.read_verified_manifest() {
        if validate_base_contents(workspace.worktree_path(), &layout.graph, &identity).is_ok()
            && validate_base_summary_cache(
                workspace.worktree_path(),
                &layout.graph,
                &layout.summary_cache,
                &identity,
            )
            .is_ok()
        {
            return prepared_base_with_reader(&layout, manifest, true);
        }
    }

    let builder_lease = layout
        .acquire_builder(false)
        .map_err(|error| Error::io("acquire Base Store builder lease", error))?
        .ok_or_else(|| Error::Lock("blocking Base builder lease returned no guard".into()))?;
    if let Ok(manifest) = layout.read_verified_manifest() {
        if validate_base_contents(workspace.worktree_path(), &layout.graph, &identity).is_ok()
            && validate_base_summary_cache(
                workspace.worktree_path(),
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
    let status = Command::new(binary)
        .arg("index")
        .current_dir(workspace.worktree_path())
        .env("GREPPY_STORE_DIR", &staging_data)
        // A published Base is not valid until every candidate has its vector;
        // never let the ordinary foreground-index lazy threshold hand this
        // build to a background process outside the publication lease.
        .env("GREPPY_LAZY_EMBED_MIN_SPANS", usize::MAX.to_string())
        .env_remove(ENV_MODE)
        .env_remove(ENV_BASE_PATH)
        .env_remove(ENV_BASE_COMMIT)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
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
        .join(greppy_core::workspace_hash(workspace.worktree_path()))
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
    validate_base_contents(workspace.worktree_path(), &staged_graph, &identity)?;
    let staged_summary_cache = build_base_summary_cache(
        workspace.worktree_path(),
        &staged_graph,
        &identity.summary_model_and_prompt_version,
    )?;
    validate_base_summary_cache(
        workspace.worktree_path(),
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

fn validate_base_contents(
    root: &Path,
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
    let project = greppy_core::project_identity(root);
    let generation = store
        .list_workspace_states()?
        .into_iter()
        .map(|state| state.graph_generation)
        .max()
        .ok_or_else(|| Error::Invalid("Base build has no workspace generation".into()))?;
    let completion_key = crate::embedding_complete_key(&project);
    let completion: Option<String> = store
        .conn()
        .query_row(
            "SELECT value FROM schema_meta WHERE key = ?1",
            [&completion_key],
            |row| row.get(0),
        )
        .ok();
    let expected_completion = format!("{generation}|{}", identity.embedding_model);
    if completion.as_deref() != Some(expected_completion.as_str()) {
        return Err(Error::Invalid(format!(
            "Base embedding generation is incomplete: expected `{expected_completion}`, got {}",
            completion.as_deref().unwrap_or("missing")
        )));
    }
    let provider_failures = store
        .list_provider_states(&project)?
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

fn build_base_summary_cache(
    root: &Path,
    graph_path: &Path,
    expected_model_key: &str,
) -> Result<PathBuf> {
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
    let expected = expected_base_summary_spans(root, graph_path)?;
    for (file_path, start_line, source, _) in &expected {
        crate::summarize_source_cached(
            &config,
            &model_key,
            Some(&cache),
            None,
            file_path,
            source,
            true,
        )
        .ok_or_else(|| {
            Error::Invalid(format!(
                "Base summary generation failed for {file_path}:{start_line}"
            ))
        })?;
    }
    let actual = cache.count()? as usize;
    if actual != expected.len() {
        return Err(Error::Invalid(format!(
            "Base summary cache is incomplete: expected {} entries, found {actual}",
            expected.len()
        )));
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
    let expected = expected_base_summary_spans(root, graph_path)?;
    if actual != expected.len() {
        return Err(Error::Invalid(format!(
            "Base summary cache is incomplete: expected {} entries, found {actual}",
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
}
