//! Index freshness: the gates that refuse to answer from a stale graph.
//!
//! Split out of `lib.rs`; `use super::*` keeps every private helper there
//! reachable, and no behaviour changes.

use super::*;

pub(crate) fn provider_policy_from_env() -> Result<ProviderPolicy> {
    let raw = match std::env::var(ENV_PROVIDER_POLICY) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(ProviderPolicy::Metadata),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::Config(format!(
                "{ENV_PROVIDER_POLICY} must be valid UTF-8"
            )));
        }
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "metadata" | "warn" | "permissive" => Ok(ProviderPolicy::Metadata),
        "require_complete" | "require-complete" | "strict" | "1" | "true" => {
            Ok(ProviderPolicy::RequireComplete)
        }
        _ => Err(Error::Config(format!(
            "{ENV_PROVIDER_POLICY} must be one of metadata or require_complete"
        ))),
    }
}

pub(crate) fn provider_policy_blocks_query(
    incomplete_providers: &[serde_json::Value],
) -> Result<bool> {
    Ok(
        provider_policy_from_env()? == ProviderPolicy::RequireComplete
            && !incomplete_providers.is_empty(),
    )
}

pub(crate) fn graph_stale_skip_json(
    store: &greppy_store::Store,
    _root: Option<&str>,
    project: &str,
    command: &str,
    freshness: serde_json::Value,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<()> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    let mut obj = serde_json::Map::new();
    obj.insert("command".into(), serde_json::json!(command));
    obj.insert("status".into(), serde_json::json!("skipped_stale_index"));
    obj.insert("project".into(), serde_json::json!(project));
    obj.insert("fresh".into(), serde_json::json!(false));
    obj.insert("freshness".into(), freshness);
    obj.insert(
        "provider_complete".into(),
        serde_json::json!(incomplete_providers.is_empty()),
    );
    obj.insert(
        "incomplete_provider_count".into(),
        serde_json::json!(incomplete_providers.len()),
    );
    obj.insert(
        "incomplete_providers".into(),
        serde_json::json!(incomplete_providers),
    );
    obj.insert("total_exact".into(), serde_json::json!(0));
    obj.insert("shown".into(), serde_json::json!(0));
    obj.insert("omitted".into(), serde_json::json!(0));
    obj.insert("truncated".into(), serde_json::json!(false));
    if let serde_json::Value::Object(extra) = extra {
        for (key, value) in extra {
            obj.insert(key, value);
        }
    }
    obj.insert(empty_collection_field.into(), serde_json::json!([]));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(obj)).map_err(|e| {
            Error::Invalid(format!("serialize stale graph JSON for {command}: {e}"))
        })?
    );
    Ok(())
}

/// Fresh-or-fallback gate for graph navigation. Indexed graph data is only
/// visible when freshness was proven; drift/unknown states trigger refresh
/// and return EX_TEMPFAIL instead of exposing stale rows.
pub(crate) fn graph_stale_gate(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    json: bool,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<Option<i32>> {
    match freshness_serve_decision(store, root, project) {
        FreshnessServe::Fresh(_) => Ok(None),
        FreshnessServe::Refuse(freshness) => {
            if json {
                graph_stale_skip_json(
                    store,
                    root,
                    project,
                    command,
                    freshness.clone(),
                    extra,
                    empty_collection_field,
                )?;
            } else {
                println!("{}", indexed_stale_skip_message(command, &freshness));
            }
            Ok(Some(freshness_refusal_exit(&freshness)))
        }
    }
}

pub(crate) fn provider_policy_graph_gate(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    command: &str,
    json: bool,
    extra: serde_json::Value,
    empty_collection_field: &str,
) -> Result<Option<i32>> {
    let incomplete_providers = incomplete_provider_json(store, project)?;
    if !provider_policy_blocks_query(&incomplete_providers)? {
        return Ok(None);
    }
    if json {
        provider_incomplete_skip_json(
            store,
            root,
            project,
            command,
            &incomplete_providers,
            extra,
            empty_collection_field,
        )?;
    } else {
        println!(
            "{}",
            provider_incomplete_skip_message(command, incomplete_providers.len())
        );
    }
    Ok(Some(1))
}

pub(crate) fn freshness_state_can_trigger_reindex(state: &str) -> bool {
    !matches!(
        state,
        "cold" | "config_error" | "failed" | "unknown" | "refreshing"
    )
}

pub(crate) fn freshness_serve_decision(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
) -> FreshnessServe {
    freshness_serve_decision_with_policy(store, root, project, true, true, true)
}

/// Heal a reindexable-stale store in-band: rebuild the graph AND (when the
/// store carried them) the embeddings + summaries at a fresh generation, then
/// re-open so the caller serves the current codebase. The edit loop mutates
/// files constantly — through greppy's own edits AND external means (git apply,
/// bash, another tool) — and every query command must reflect those changes or
/// the agent gets stale/empty answers and abandons greppy (forensics
/// 2026-07-18). Genuinely un-reindexable states (cold/failed) are left for the
/// stale gate to refuse. Best-effort: a failed reindex leaves the old store,
/// and the gate then decides.
pub(crate) fn maybe_reindex_stale(
    store: &mut greppy_store::Store,
    root: Option<&str>,
) -> Result<()> {
    maybe_reindex_stale_with_capability(store, root, true, true)
}

pub(crate) fn maybe_reindex_stale_semantic(
    store: &mut greppy_store::Store,
    root: Option<&str>,
    can_rebuild_vectors: bool,
) -> Result<()> {
    maybe_reindex_stale_with_capability(store, root, false, can_rebuild_vectors)
}

fn maybe_reindex_stale_with_capability(
    store: &mut greppy_store::Store,
    root: Option<&str>,
    structural_only: bool,
    allow_auto_reindex: bool,
) -> Result<()> {
    // An explicit auto-reindex opt-out must fall through to the fail-closed
    // stale gate. In particular, do not wait on an active writer that the
    // caller has said must not be joined for automatic healing.
    if !auto_reindex_enabled() || !allow_auto_reindex {
        return Ok(());
    }
    let project = project_for(root)?;
    let freshness = nav_freshness_json(store, root, &project);
    if freshness_is_reindexable_stale(&freshness) {
        if structural_only {
            let effective_root = resolve_root(root)?;
            wait_for_index_publication(root, &effective_root, "structural-workspace-drift")?;
            if let Ok(fresh) = open_default_store_query_writer(root) {
                *store = fresh;
            }
            return Ok(());
        }
        let rebuilt =
            freshness_within_inline_drift_cap(root, &freshness) && try_auto_reindex_inline(root);
        if !rebuilt {
            let effective_root = resolve_root(root)?;
            wait_for_index_publication(root, &effective_root, "workspace-drift")?;
        }
        if let Ok(fresh) = open_default_store_query_writer(root) {
            *store = fresh;
        }
    }
    Ok(())
}

/// The index is stale AND the drift is one an automatic reindex can heal
/// (workspace/content drift or a scope-stable version bump), not a cold or
/// broken store. Structural queries own publication even for large drift;
/// semantic callers retain the bounded inline/background refresh policy.
pub(crate) fn freshness_is_reindexable_stale(freshness: &serde_json::Value) -> bool {
    if freshness_json_is_fresh(freshness) {
        return false;
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    if !freshness_state_can_trigger_reindex(state) {
        return false;
    }
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|reasons| {
            reasons
                .iter()
                .filter_map(serde_json::Value::as_str)
                .any(|reason| reason.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        return version_drift_is_scope_stable(freshness);
    }
    if metadata_only_fingerprint_drift(freshness) {
        return false;
    }
    freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .is_some()
}

fn freshness_within_inline_drift_cap(root: Option<&str>, freshness: &serde_json::Value) -> bool {
    freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|count| {
            count as usize <= AUTO_REINDEX_MAX_FILES
                && freshness_changed_bytes(root, freshness)
                    .is_some_and(|bytes| bytes <= 8 * 1024 * 1024)
        })
}

pub(crate) fn try_refresh_metadata_only_fingerprint(
    root: Option<&str>,
    freshness: &serde_json::Value,
) -> Option<serde_json::Value> {
    if !metadata_only_fingerprint_drift(freshness) {
        return None;
    }
    let effective_root = resolve_root(root).ok()?;
    let overrides = discover_overrides_from_env().ok()?;
    let store_path = workspace_locator::store_path(&effective_root);
    let _writer = greppy_freshness::try_acquire(&store_path).ok()?;
    let mut store =
        greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::query_writer())
            .ok()?;
    let fingerprint = greppy_core::GitFingerprint::capture(&effective_root);
    if !greppy_freshness::refresh_fingerprint_metadata(
        &mut store,
        &fingerprint,
        NAV_FRESHNESS_BUDGET,
        &overrides,
    )
    .ok()?
    {
        return None;
    }

    let mut refreshed = freshness.clone();
    let object = refreshed.as_object_mut()?;
    object.insert("fresh".into(), serde_json::Value::Bool(true));
    object.insert("state".into(), serde_json::Value::String("fresh".into()));
    object.insert("reasons".into(), serde_json::Value::Array(Vec::new()));
    Some(refreshed)
}

pub(crate) fn freshness_serve_decision_with_policy(
    store: &greppy_store::Store,
    root: Option<&str>,
    project: &str,
    allow_auto_reindex: bool,
    _warn_on_stale: bool,
    structural_only: bool,
) -> FreshnessServe {
    let refresh_cause = if structural_only {
        "structural-workspace-drift"
    } else {
        "workspace-drift"
    };
    let writer_active = workspace_writer_active(root);
    // A writer may be publishing metadata-only drift while the indexed file
    // contents remain exactly valid. Bypass the freshness stamp so serving
    // under contention requires a current inventory proof; changed or
    // unverifiable contents remain fail-closed below.
    let freshness = if writer_active {
        nav_freshness_json_uncached(store, root, project)
    } else {
        nav_freshness_json(store, root, project)
    };
    if freshness_json_is_fresh(&freshness) {
        return FreshnessServe::Fresh(freshness);
    }
    if writer_active {
        return FreshnessServe::Refuse(refresh_state(freshness, true));
    }
    let state = freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    // Unknown is not evidence of drift. In particular, a budget-exhausted
    // inventory walk must not launch a full reindex that can replace the DB
    // containing expand packs created by the preceding query.
    if !freshness_state_can_trigger_reindex(state) {
        return FreshnessServe::Refuse(freshness);
    }
    let scope_or_version_drift = freshness
        .get("reasons")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|rs| {
            rs.iter()
                .filter_map(serde_json::Value::as_str)
                .any(|r| r.contains("indexer version/scope"))
        });
    if scope_or_version_drift {
        if allow_auto_reindex && auto_reindex_enabled() && version_drift_is_scope_stable(&freshness)
        {
            let started = spawn_background_index(root, refresh_cause);
            return FreshnessServe::Refuse(refresh_state(
                freshness,
                started || workspace_writer_active(root),
            ));
        }
        return FreshnessServe::Refuse(freshness);
    }

    // A commit can change only HEAD after the exact source contents were
    // already indexed. The inventory diff above proves there are zero stale
    // files, so refresh just the fingerprint instead of rebuilding the graph
    // and every embedding at a new generation.
    if allow_auto_reindex && auto_reindex_enabled() {
        if let Some(refreshed) = try_refresh_metadata_only_fingerprint(root, &freshness) {
            return FreshnessServe::Fresh(refreshed);
        }
    }

    let stale_file_count = freshness
        .get("stale_file_count")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    let small_enough = stale_file_count.is_some_and(|count| {
        count <= AUTO_REINDEX_MAX_FILES
            && freshness_changed_bytes(root, &freshness)
                .is_some_and(|bytes| bytes <= 8 * 1024 * 1024)
    });
    if allow_auto_reindex && auto_reindex_enabled() && small_enough {
        let rebuilt = !structural_only && try_auto_reindex_inline(root);
        let writer_active = workspace_writer_active(root);
        let started = if rebuilt || writer_active {
            false
        } else {
            spawn_background_index(root, refresh_cause)
        };
        return FreshnessServe::Refuse(refresh_state(
            freshness,
            rebuilt || started || writer_active || workspace_writer_active(root),
        ));
    }
    if allow_auto_reindex && auto_reindex_enabled() {
        let started = spawn_background_index(root, refresh_cause);
        return FreshnessServe::Refuse(refresh_state(
            freshness,
            started || workspace_writer_active(root),
        ));
    }
    FreshnessServe::Refuse(freshness)
}

pub(crate) fn freshness_changed_bytes(
    root: Option<&str>,
    freshness: &serde_json::Value,
) -> Option<u64> {
    let root = resolve_root(root).ok()?;
    let paths = freshness.get("changed_paths")?.as_array()?;
    let mut bytes = 0u64;
    for path in paths {
        let path = path.as_str()?;
        match std::fs::metadata(root.join(path)) {
            Ok(metadata) => bytes = bytes.saturating_add(metadata.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return None,
        }
    }
    Some(bytes)
}

pub(crate) fn freshness_refusal_exit(freshness: &serde_json::Value) -> i32 {
    match freshness
        .get("state")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
    {
        "refreshing" | "drift" | "unknown" => EXIT_TEMPFAIL as i32,
        _ => 1,
    }
}

pub(crate) fn workspace_writer_active(root: Option<&str>) -> bool {
    let Ok(root) = resolve_root(root) else {
        return false;
    };
    background_job_writer_active(&root)
}

pub(crate) fn auto_reindex_inline_allowed(
    had_vectors: bool,
    indexed_files: i64,
    _overlay: bool,
) -> bool {
    !had_vectors && (0..=AUTO_REINDEX_INLINE_MAX_INDEXED_FILES).contains(&indexed_files)
}

/// Build a genuinely bounded small-drift refresh through the same
/// temp-snapshot publication boundary as an explicit `index`. Vector-backed
/// or large full stores return false so the caller starts one observable
/// background refresh instead of hiding model loading or a full repository
/// rebuild inside a navigation command.
pub(crate) fn try_auto_reindex_inline(root: Option<&str>) -> bool {
    let Ok(effective_root) = resolve_root(root) else {
        return false;
    };
    let Ok(project) = project_for(root) else {
        return false;
    };
    let Ok(overrides) = discover_overrides_from_env() else {
        return false;
    };
    let store_path = workspace_locator::store_path(&effective_root);
    let Ok(Some(_lifecycle)) = greppy_core::cache::acquire_workspace_lifecycle(
        &effective_root,
        greppy_core::cache::LockMode::Shared,
        false,
    ) else {
        return false;
    };
    let _lock = match greppy_freshness::try_acquire(&store_path) {
        Ok(lock) => lock,
        _ => return false, // another writer is active: refuse this snapshot
    };
    let overlay = crate::store_cow::overlay_spec_live(&effective_root)
        .ok()
        .flatten();
    let store = match overlay.as_ref() {
        Some(overlay) => greppy_store::Store::open_overlay_read_only(
            &overlay.base_path,
            &store_path,
            &overlay.visibility,
        ),
        None => greppy_store::Store::open_with(&store_path, greppy_store::OpenOptions::read_only()),
    };
    let Ok(store) = store else {
        return false;
    };
    // Remember whether this store served code-span vectors BEFORE the
    // reindex bumps the generation: an inline graph-only reindex would
    // otherwise strand every existing vector row on the old generation and
    // silently degrade `context`/`semantic-search` until a manual
    // `grep index` run (the owner's "gains" path dying quietly).
    let had_vectors = !store
        .vector_model_ids(&project)
        .unwrap_or_default()
        .is_empty();
    let indexed_files = store.file_count(&project).unwrap_or(i64::MAX);
    if !auto_reindex_inline_allowed(had_vectors, indexed_files, overlay.is_some()) {
        return false;
    }
    drop(store);
    let options = greppy_indexer::IndexOptions {
        discover_overrides: overrides,
        only_paths: None,
    };
    if let Some(overlay) = overlay.as_ref() {
        crate::indexing::index_overlay_snapshot(
            &store_path,
            &effective_root,
            &project,
            overlay,
            None,
            &options,
            false,
            None,
        )
        .map(|_| true)
        .unwrap_or(false)
    } else {
        index_atomic_snapshot(
            &store_path,
            &effective_root,
            &project,
            None,
            &options,
            false,
            None,
        )
        .map(|snapshot| snapshot.index.is_clean())
        .unwrap_or(false)
    }
}

pub(crate) fn freshness_json_is_fresh(freshness: &serde_json::Value) -> bool {
    freshness
        .get("fresh")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn open_default_store(root: Option<&str>) -> Result<greppy_store::Store> {
    // The graph DB lives under the platform locator, never at
    // `<cwd>/.greppy/graph.db`. When no
    // `--root` is given we detect the repo root by walking up for a
    // marker, so a query from a subdirectory targets the same store the
    // indexer wrote from the repo root (instead of opening an empty
    // store under the subdir's hash and exiting 73).
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    // RV-007: tighten the store dir + DB file permissions on every open.
    // This is a no-op when the store doesn't exist yet (read paths before
    // any `greppy index` would have failed to open the store anyway).
    if let Some(parent) = path.parent() {
        let _ = workspace_locator::ensure_store_dir(parent);
    }
    if let Some((base_path, base_commit)) = crate::store_cow::overlay_environment(&effective_root)?
    {
        greppy_core::cache::ensure_workspace_store(&effective_root).map_err(|error| {
            Error::io(
                format!(
                    "create private Delta Store for {}",
                    effective_root.display()
                ),
                error,
            )
        })?;
        if !path.exists() {
            drop(greppy_store::Store::open(&path)?);
        }
        let delta = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())?;
        let visibility = crate::store_cow::visibility_for_open_connection(
            &effective_root,
            &base_commit,
            delta.conn(),
        )?;
        let store = delta.attach_overlay(&base_path, &visibility)?;
        let _ = workspace_locator::ensure_db_mode(&path);
        if let Some(store_dir) = path.parent() {
            workspace_locator::touch_lastused(store_dir);
        }
        return Ok(store);
    }
    // Forensics F4: a query against a repo that was never indexed used to
    // open a non-existent DB, fail deep in SQLite, and exit 73 (EXIT_IO)
    // with NOTHING on stdout/stderr — the agent just saw an empty result and
    // a bare non-zero code, with no hint that the fix is `greppy index`.
    //
    // Auto-index on first use, but never hide an unbounded repository walk
    // inside a navigation command. Start the ordinary detached indexer and
    // join it for at most the same two-second window used by stale queries.
    // Small repositories still feel immediate; large repositories return a
    // retryable result with a stable status surface instead of appearing hung.
    // Gated behind GREPPY_AUTO_REINDEX so explicit opt-out keeps the old error.
    // (Query commands only — the grep passthrough path never reaches here,
    // so the byte-exact passthrough contract is untouched.)
    if !path.exists() {
        if auto_reindex_enabled() {
            wait_for_first_use_index(root, &effective_root)?;
        } else {
            let shown_root = root.unwrap_or(".");
            eprintln!(
                "greppy: no index for {} — run `greppy index {}` first",
                effective_root.display(),
                shown_root
            );
            return Err(Error::Invalid(format!(
                "no index for {}; run `greppy index {}` first",
                effective_root.display(),
                shown_root
            )));
        }
    }
    // Query commands are READ-ONLY: open read-only so they skip both
    // `migrate()` and the O(db-size) `integrity_check` that a read-write open
    // runs. Those belong on the writer (`greppy index`); paying them on every
    // query open made who-calls/search take seconds on a real repo
    // (the token-efficiency benchmark's latency culprit). Readers tolerate
    // whatever schema the DB has.
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::read_only())?;
    // Command-output packs can create a database before any graph is published.
    // Its existence alone is not a completed first index. Bootstrap the graph
    // exactly as for a missing database instead of serving false empty results.
    if auto_reindex_enabled()
        && store
            .get_workspace_state(effective_root.to_string_lossy().as_ref())?
            .is_none()
    {
        drop(store);
        wait_for_first_use_index(root, &effective_root)?;
        return open_default_store(root);
    }
    let _ = workspace_locator::ensure_db_mode(&path);
    // Feature B: record that this store was just used to serve a query.
    // A read-only open never bumps graph.db's mtime, so a dedicated
    // `.lastused` marker is what keeps a frequently-queried store from
    // being evicted by `cleanup_stale_stores`. Best-effort — a failed
    // touch never fails the query.
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    // O5 session prewarm: the first graph command of an agent session nudges
    // the embed daemon (with an async model load) so a following `context`
    // query hits a warm model instead of paying the cold start. Guarded to
    // fire only when semantic search is actually in play — env model
    // configured AND this store holds vectors — because prewarming a model
    // nobody will query would hold GPU memory for a TTL for nothing.
    #[cfg(any(unix, windows))]
    {
        let has_vectors = project_for(root)
            .ok()
            .and_then(|p| store.vector_model_ids(&p).ok())
            .is_some_and(|models| !models.is_empty());
        if has_vectors {
            let no_args = EmbeddingCliArgs {
                device: None,
                no_gpu: false,
            };
            if let Ok(Some(cfg)) = embedding_config_optional(no_args) {
                let key = embedding_query_cache_key(&cfg);
                embed_daemon::prewarm_from_env(&cfg, &key);
            }
        }
    }
    Ok(store)
}

#[derive(Debug, PartialEq, Eq)]
enum FirstUseIndexObservation {
    Pending,
    Published,
    Failed(String),
}

fn observe_first_use_index(
    job: Option<&serde_json::Value>,
    snapshot_ready: bool,
    owner_active: bool,
) -> FirstUseIndexObservation {
    // The portable OS writer lock is the ownership authority. A foreground
    // writer may have inherited an old background record and has not yet
    // replaced or removed it, so historical state cannot overrule a verified
    // current owner.
    if owner_active {
        return FirstUseIndexObservation::Pending;
    }
    // Publication is authoritative after the owner releases its lock. A
    // foreground writer is allowed to publish without owning the historical
    // background record, which may therefore remain stale.
    if snapshot_ready {
        return FirstUseIndexObservation::Published;
    }
    if let Some(job) = job {
        let state = job
            .get("state")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        if state == "failed" {
            return FirstUseIndexObservation::Failed(
                job.get("last_error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("no error was recorded")
                    .to_owned(),
            );
        }
        return FirstUseIndexObservation::Failed(format!(
            "job owner exited before publishing a snapshot (last state: {state})"
        ));
    }
    FirstUseIndexObservation::Failed(
        "job ended before publishing a snapshot or recording an error".into(),
    )
}

fn published_graph_generation(effective_root: &std::path::Path) -> Option<u64> {
    greppy_store::Store::open_with(
        &workspace_locator::store_path(effective_root),
        greppy_store::OpenOptions::read_only(),
    )
    .ok()
    .and_then(|store| {
        store
            .get_workspace_state(effective_root.to_string_lossy().as_ref())
            .ok()
            .flatten()
    })
    .map(|state| state.graph_generation)
}

fn publication_advanced(baseline: Option<u64>, published: Option<u64>) -> bool {
    published.is_some() && published != baseline
}

/// A structural query owns completion of the graph publication it starts. There
/// is no elapsed-time cutoff: a slow but live extraction remains attached, while
/// a failed or dead owner terminates immediately with the recorded cause. Process
/// interruption still cancels the waiting query; the detached indexer keeps its
/// existing durable contract.
fn wait_for_first_use_index(root: Option<&str>, effective_root: &std::path::Path) -> Result<()> {
    wait_for_index_publication(root, effective_root, "first-use")
}

fn wait_for_index_publication(
    root: Option<&str>,
    effective_root: &std::path::Path,
    cause: &str,
) -> Result<()> {
    let baseline_generation = published_graph_generation(effective_root);
    let mut launch = spawn_background_job_handle(root, cause, "index", None).ok_or_else(|| {
        let detail = read_background_job(&background_job_path(effective_root))
            .and_then(|job| {
                job.get("last_error")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "the index process could not be started".into());
        Error::Index(format!(
            "structural index failed for {}: {detail}",
            effective_root.display()
        ))
    })?;
    loop {
        let owner_active = launch.owner_is_active().map_err(|error| {
            Error::io(
                format!(
                    "observe structural index owner for {}",
                    effective_root.display()
                ),
                error,
            )
        })?;
        let job = read_background_job(launch.path());
        // Never reopen SQLite while its verified writer is active. Once the
        // lock is released, publication outranks a historical job record,
        // including a stale failed/nonterminal record left by another owner.
        let snapshot_ready = !owner_active
            && publication_advanced(
                baseline_generation,
                published_graph_generation(effective_root),
            );
        match observe_first_use_index(job.as_ref(), snapshot_ready, owner_active) {
            FirstUseIndexObservation::Pending => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            FirstUseIndexObservation::Published => {
                if let BackgroundJobLaunch::Owned { child, .. } = &mut launch {
                    let _ = child.wait();
                }
                return Ok(());
            }
            FirstUseIndexObservation::Failed(detail) => {
                return Err(Error::Index(format!(
                    "structural index failed for {}: {detail}",
                    effective_root.display()
                )));
            }
        }
    }
}

pub(crate) fn open_default_store_query_writer(root: Option<&str>) -> Result<greppy_store::Store> {
    open_default_store_writer(root, true)
}

/// Open the small writable evidence/continuation surface without forcing a
/// graph build. Exact filesystem reads must remain available before the first
/// index; their pagination records are not graph-query evidence.
pub(crate) fn open_default_store_pack_writer(root: Option<&str>) -> Result<greppy_store::Store> {
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    if let Some(parent) = path.parent() {
        workspace_locator::ensure_store_dir(parent)
            .map_err(|error| Error::io("create continuation pack store", error))?;
    }
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())?;
    let _ = workspace_locator::ensure_db_mode(&path);
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    Ok(store)
}

fn open_default_store_writer(
    root: Option<&str>,
    require_existing_index: bool,
) -> Result<greppy_store::Store> {
    let effective_root = resolve_root(root)?;
    let path = workspace_locator::store_path(&effective_root);
    if let Some(overlay) = crate::store_cow::overlay_spec(&effective_root)? {
        greppy_core::cache::ensure_workspace_store(&effective_root).map_err(|error| {
            Error::io(
                format!(
                    "create private Delta Store for {}",
                    effective_root.display()
                ),
                error,
            )
        })?;
        if !path.exists() {
            drop(greppy_store::Store::open(&path)?);
        }
        return greppy_store::Store::open_overlay(&overlay.base_path, &path, &overlay.visibility)
            .map_err(Into::into);
    }
    if require_existing_index && !path.exists() {
        // Reuse the normal query open to trigger the existing first-use
        // auto-index/error path, then reopen writable for the evidence write.
        drop(open_default_store(root)?);
    }
    if let Some(parent) = path.parent() {
        let _ = workspace_locator::ensure_store_dir(parent);
    }
    let store = greppy_store::Store::open_with(&path, greppy_store::OpenOptions::query_writer())?;
    let _ = workspace_locator::ensure_db_mode(&path);
    if let Some(store_dir) = path.parent() {
        workspace_locator::touch_lastused(store_dir);
    }
    Ok(store)
}

pub(crate) fn cleanup_expired_legacy_entries(
    current: Option<&std::path::Path>,
    ttl: std::time::Duration,
) {
    if ttl.is_zero() {
        return;
    }
    let now = unix_now_secs_cli();
    for entry in verified_legacy_cache_entries() {
        if current == Some(entry.root.as_path()) || entry.locked {
            continue;
        }
        if now.saturating_sub(entry.last_used_unix_secs) > ttl.as_secs() {
            let _ = remove_verified_legacy_entry(&entry);
        }
    }
}

/// Resume only legacy trash entries whose name, SQLite header, schema and
/// workspace hash all prove that Greppy created them. Unknown trash is left
/// untouched and remains visible as unmanaged cache data.
pub(crate) fn cleanup_verified_legacy_trash() {
    let Ok(entries) = std::fs::read_dir(greppy_core::cache::trash_root()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(rest) = name.strip_prefix("legacy-") else {
            continue;
        };
        let Some(hash) = rest.get(..16) else {
            continue;
        };
        if !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            || rest.as_bytes().get(16) != Some(&b'-')
        {
            continue;
        }
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            continue;
        }
        let graph = path.join("graph.db");
        if !sqlite_header_is_valid(&graph) {
            continue;
        }
        let Ok(connection) = rusqlite::Connection::open_with_flags(
            &graph,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) else {
            continue;
        };
        let schema_valid = connection
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .is_some();
        let workspace_valid = connection
            .query_row(
                "SELECT root_path FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .is_some_and(|root| {
                greppy_core::workspace::workspace_hash(std::path::Path::new(&root))
                    .eq_ignore_ascii_case(hash)
            });
        drop(connection);
        if schema_valid && workspace_valid {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

pub(crate) fn cleanup_stale_snapshot_artifacts(
    active_path: &std::path::Path,
    include_quarantine: bool,
) -> Result<usize> {
    let Some(parent) = active_path.parent() else {
        return Ok(0);
    };
    let Some(file_name) = active_path.file_name().and_then(|s| s.to_str()) else {
        return Ok(0);
    };
    let next_prefix = format!("{file_name}.next.");
    let corrupt_prefix = format!("{file_name}.corrupt.");
    let previous = format!("{file_name}.prev");
    let previous_sidecar_prefix = format!("{previous}-");
    let mut removed = 0usize;
    let entries = match std::fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(Error::io(format!("scan {}", parent.display()), e)),
    };
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(format!("scan {}", parent.display()), e))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let managed = name.starts_with(&next_prefix)
            || name == previous
            || name.starts_with(&previous_sidecar_prefix)
            || (name.starts_with(".index.job.") && name.ends_with(".tmp"))
            || (include_quarantine && name.starts_with(&corrupt_prefix));
        if !managed {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::io(
                    format!("remove stale temp {}", path.display()),
                    e,
                ))
            }
        }
    }
    if removed > 0 {
        sync_parent_dir(active_path)?;
    }
    Ok(removed)
}

pub(crate) fn cleanup_sqlite_family(path: &std::path::Path) -> Result<()> {
    remove_file_if_exists(path)?;
    cleanup_sqlite_sidecars(path)
}

pub(crate) fn cleanup_sqlite_sidecars(path: &std::path::Path) -> Result<()> {
    remove_file_if_exists(&sqlite_sidecar(path, "-wal"))?;
    remove_file_if_exists(&sqlite_sidecar(path, "-shm"))
}

#[cfg(test)]
mod refresh_wait_tests {
    use super::{
        background_refresh_is_pending, observe_first_use_index, publication_advanced,
        FirstUseIndexObservation,
    };

    #[test]
    fn stale_snapshot_is_not_a_new_structural_publication() {
        assert!(publication_advanced(None, Some(1)));
        assert!(publication_advanced(Some(7), Some(8)));
        assert!(!publication_advanced(Some(7), Some(7)));
        assert!(!publication_advanced(Some(7), None));
    }

    #[test]
    fn failed_refresh_does_not_hide_behind_the_old_snapshot() {
        let failed = serde_json::json!({
            "state": "failed",
            "last_error": "fixture structural extraction failed",
        });
        assert_eq!(
            observe_first_use_index(Some(&failed), publication_advanced(Some(7), Some(7)), false,),
            FirstUseIndexObservation::Failed("fixture structural extraction failed".into())
        );
        assert_eq!(
            observe_first_use_index(Some(&failed), publication_advanced(Some(7), Some(8)), false,),
            FirstUseIndexObservation::Published,
            "a newer atomic publication remains authoritative over a stale job record"
        );
    }

    #[test]
    fn launch_record_without_writer_lock_is_still_pending() {
        assert!(background_refresh_is_pending(&serde_json::json!({
            "state": "launching",
            "pid": null
        })));
    }

    #[test]
    fn failed_refresh_record_is_not_publication() {
        assert!(!background_refresh_is_pending(&serde_json::json!({
            "state": "failed",
            "pid": null,
            "last_error": "fixture"
        })));
    }

    #[test]
    fn healthy_slow_first_use_remains_pending_until_publication() {
        let job = serde_json::json!({
            "state": "loading_model",
            "pid": null,
            "completed_spans": 0,
            "total_spans": 2
        });
        assert_eq!(
            observe_first_use_index(Some(&job), false, true),
            FirstUseIndexObservation::Pending
        );
        assert_eq!(
            observe_first_use_index(None, true, false),
            FirstUseIndexObservation::Published
        );
    }

    #[test]
    fn failed_and_dead_first_use_jobs_keep_their_failure_contract() {
        let failed = serde_json::json!({
            "state": "failed",
            "pid": null,
            "last_error": "fixture model load failed"
        });
        assert_eq!(
            observe_first_use_index(Some(&failed), false, false),
            FirstUseIndexObservation::Failed("fixture model load failed".into())
        );
        let dead = serde_json::json!({"state": "indexing", "pid": u32::MAX});
        assert!(matches!(
            observe_first_use_index(Some(&dead), false, false),
            FirstUseIndexObservation::Failed(detail)
                if detail.contains("owner exited") && detail.contains("indexing")
        ));
        let abandoned_launch = serde_json::json!({"state": "launching", "pid": null});
        assert!(matches!(
            observe_first_use_index(Some(&abandoned_launch), false, false),
            FirstUseIndexObservation::Failed(detail)
                if detail.contains("owner exited") && detail.contains("launching")
        ));
    }

    #[test]
    fn foreground_writer_and_publication_override_stale_job_state() {
        for stale in [
            serde_json::json!({
                "state": "failed",
                "pid": 7,
                "last_error": "historical failure"
            }),
            serde_json::json!({"state": "indexing", "pid": 7}),
        ] {
            assert_eq!(
                observe_first_use_index(Some(&stale), false, true),
                FirstUseIndexObservation::Pending,
                "verified foreground writer owns progress despite stale record"
            );
            assert_eq!(
                observe_first_use_index(Some(&stale), true, false),
                FirstUseIndexObservation::Published,
                "published snapshot wins after foreground writer releases"
            );
        }
    }

    #[test]
    fn second_caller_waits_only_for_a_verified_owner() {
        let job = serde_json::json!({
            "state": "loading_model",
            "pid": std::process::id()
        });
        assert_eq!(
            observe_first_use_index(Some(&job), false, true),
            FirstUseIndexObservation::Pending
        );
        assert!(matches!(
            observe_first_use_index(Some(&job), false, false),
            FirstUseIndexObservation::Failed(detail)
                if detail.contains("owner exited")
        ));
    }
}
